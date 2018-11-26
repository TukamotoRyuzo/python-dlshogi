import argparse
import logging
import random

from keras.optimizers import SGD

import pydlshogi.common as cmn
from pydlshogi.network.value_bn import ValueNetwork

from tqdm import tqdm

from train_tools import create_mini_batch, load_kifu_data

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

v_net = ValueNetwork()
v_net.compile(
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss='binary_crossentropy',
    metrics=['accuracy'])
v_net.summary()

# Init/Resume
if args.initmodel:
    logging.info('Load model from %s', args.initmodel)
    v_net.load_weights(args.initmodel)

positions_train, positions_test = load_kifu_data(args)

num_classes = 9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM


class Status:

    def __init__(self):
        self.loss_sum = 0
        self.acc_sum = 0
        self.iter_num = 0
        self.loss = 0
        self.acc = 0

    def update(self, results):
        self.loss_sum += results[0]
        self.acc_sum += results[1]
        self.iter_num += 1

    def calc_mean(self):
        self.loss = self.loss_sum / self.iter_num
        self.acc = self.acc_sum / self.iter_num

    def reset(self):
        self.loss_sum = 0
        self.acc_sum = 0
        self.iter_num = 0


# train & validation
logging.info('start training')
train_size = len(positions_train)
for e in range(args.epoch):
    # train
    train_status = Status()

    positions_train_shuffled = random.sample(positions_train, train_size)

    interval_status = Status()
    for batch_pos in tqdm(range(0, train_size - args.batchsize, args.batchsize)):
        x, _, t2 = create_mini_batch(positions_train_shuffled, batch_pos,
                                     args.batchsize)
        results = v_net.train_on_batch(x, t2)
        train_status.update(results)
        interval_status.update(results)

        # print train loss and accuracy
        if train_status.iter_num % args.eval_interval == 0:
            interval_status.calc_mean()
            logging.info(
                'epoch = %s, iteration = %s, interval loss = %s, interval accuracy = %s',
                e + 1,
                train_status.iter_num,
                "{:.4f}".format(interval_status.loss),
                "{:.4f}".format(interval_status.acc))  # yapf: disable
            interval_status.reset()

    end_pos = batch_pos + args.batchsize
    remain_size = train_size - end_pos
    if remain_size > 0:
        x, _, t2 = create_mini_batch(positions_train_shuffled, end_pos, remain_size)
        results = v_net.train_on_batch(x, t2)
        train_status.update(results)

    # validate test data
    logging.info('validate test data')
    test_status = Status()
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, _, t2 = create_mini_batch(positions_test, i, args.batchsize)
        result = v_net.test_on_batch(x, t2)
        test_status.update(result)

    test_status.calc_mean()
    logging.info(
        'validation finished: epoch = %s, iteration = %s, test loss = %s, test accuracy = %s',
        e + 1,
        test_status.iter_num,
        "{:.4f}".format(test_status.loss),
        "{:.4f}".format(test_status.acc))  # yapf: disable

    logging.info("epoch %s finished", e + 1)
    logging.info('save the model')
    v_net.save_weights('value_bn_epoch{}.h5'.format(e + 1))

    v_net.optimizer.lr = v_net.optimizer.lr * 0.92
