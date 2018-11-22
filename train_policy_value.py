import argparse
import logging
import random

from keras.metrics import binary_accuracy, categorical_accuracy
from keras.optimizers import SGD

import pydlshogi.common as cmn
from pydlshogi.network.policy_value import PolicyValueNetwork

from tqdm import tqdm

from train_tools import Status, create_mini_batch, load_kifu_data

parser = argparse.ArgumentParser()
# yapf: disable
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_policy_value', help='model file name')
parser.add_argument('--state', type=str, default='model/state_policy_value', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--eval_interval', '-i', type=int, default=1000, help='eval interval')
# yapf: enable
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    filename=args.log,
    level=logging.DEBUG)

pv_net = PolicyValueNetwork()
pv_net.compile(
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss={
        'policy_out': 'categorical_crossentropy',
        'value_out': 'binary_crossentropy'
    },
    metrics={
        'policy_out': categorical_accuracy,
        'value_out': binary_accuracy
    },
    loss_weights={
        'policy_out': 0.5,
        'value_out': 1.5
    })
pv_net.summary()

# Init/Resume
if args.initmodel:
    logging.info('Load model from %s', args.initmodel)
    pv_net.load_weights(args.initmodel)

positions_train, positions_test = load_kifu_data(args)

num_classes = 9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM

# train & validation
logging.info('start training')
train_size = len(positions_train)
for e in range(args.epoch):
    # train
    train_status = Status()

    positions_train_shuffled = random.sample(positions_train, train_size)

    interval_status = Status()
    for batch_pos in tqdm(range(0, train_size - args.batchsize, args.batchsize)):
        x, t1, t2 = create_mini_batch(positions_train_shuffled, batch_pos,
                                      args.batchsize)
        results = pv_net.train_on_batch(x, {'policy_out': t1, 'value_out': t2})
        train_status.update(results)
        interval_status.update(results)

        # print train loss and accuracy
        if train_status.iter_num % args.eval_interval == 0:
            interval_status.calc_mean()
            logging.info(
                'epoch = %s, iteration = %s, interval loss = %s(p: %s, v: %s), interval accuracy = p: %s, v: %s',
                e + 1,
                train_status.iter_num,
                "{:.4f}".format(interval_status.total_loss),
                "{:.4f}".format(interval_status.policy_loss),
                "{:.4f}".format(interval_status.value_loss),
                "{:.4f}".format(interval_status.policy_acc),
                "{:.4f}".format(interval_status.value_acc))  # yapf: disable
            interval_status.reset()

    end_pos = batch_pos + args.batchsize
    remain_size = train_size - end_pos
    if remain_size > 0:
        x, t1, t2 = create_mini_batch(positions_train_shuffled, end_pos, remain_size)
        results = pv_net.train_on_batch(x, {'policy_out': t1, 'value_out': t2})
        train_status.update(results)

    # validate test data
    logging.info('validate test data')
    test_status = Status()
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, t1, t2 = create_mini_batch(positions_test, i, args.batchsize)
        result = pv_net.test_on_batch(x, {'policy_out': t1, 'value_out': t2})
        test_status.update(result)

    test_status.calc_mean()
    logging.info(
        'validation finished: epoch = %s, iteration = %s, test loss = %s(p: %s, v: %s), test accuracy = p: %s, v: %s',
        e + 1,
        test_status.iter_num,
        "{:.4f}".format(test_status.total_loss),
        "{:.4f}".format(test_status.policy_loss),
        "{:.4f}".format(test_status.value_loss),
        "{:.4f}".format(test_status.policy_acc),
        "{:.4f}".format(test_status.value_acc))  # yapf: disable

    logging.info("epoch %s finished", e + 1)
    logging.info('save the model')
    pv_net.save_weights('init_valuepolicy_epoch{}.h5'.format(e + 1))

    pv_net.optimizer.lr = pv_net.optimizer.lr * 0.92
