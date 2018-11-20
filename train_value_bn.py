import argparse
import logging
import os
import pickle
import random
import re

from keras.optimizers import SGD

import numpy as np

import pydlshogi.features as fts
from pydlshogi.network.value import ValueNetwork
from pydlshogi.read_kifu import read_kifu

from tqdm import tqdm

parser = argparse.ArgumentParser()
# yapf: disable
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_value', help='model file name')
parser.add_argument('--state', type=str, default='model/state_value', help='state file name')
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
    SGD(lr=0.01, momentum=0.9, nesterov=True),
    'binary_crossentropy',
    metrics=['accuracy'])
v_net.summary()

# Init/Resume
if args.initmodel:
    logging.info('Load model from {}'.format(args.initmodel))
    v_net.load_weights(args.initmodel)

logging.info('read kifu start')
# 保存済みのpickleファイルがある場合、pickleファイルを読み込む
# train date
train_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_train) + '.pickle'
if os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'rb') as f:
        positions_train = pickle.load(f)
    logging.info('load train pickle')
else:
    positions_train = read_kifu(args.kifulist_train)

# test data
test_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_test) + '.pickle'
if os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'rb') as f:
        positions_test = pickle.load(f)
    logging.info('load test pickle')
else:
    positions_test = read_kifu(args.kifulist_test)

# 保存済みのpickleがない場合、pickleファイルを保存する
if not os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'wb') as f:
        pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save train pickle')
if not os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'wb') as f:
        pickle.dump(positions_test, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save test pickle')
logging.info('read kifu end')

logging.info('train position num = %s', len(positions_train))
logging.info('test position num = %s', len(positions_test))


# mini batch
def create_mini_batch(positions, batch_head, batch_size):
    mini_batch_data = []
    mini_batch_win = []
    for b in range(batch_size):
        features, _, win = fts.make_features(positions[batch_head + b])
        mini_batch_data.append(features)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float),
            np.array(mini_batch_win, dtype=np.int).reshape((-1, 1)))


# train
logging.info('start training')
train_size = len(positions_train)
for e in range(args.epoch):
    # train
    train_loss_sum = 0
    train_acc_sum = 0
    train_itr = 0

    positions_train_shuffled = random.sample(positions_train, train_size)

    interval_loss_sum = 0
    interval_acc_sum = 0
    interval_itr = 0
    for batch_pos in tqdm(range(0, train_size - args.batchsize, args.batchsize)):
        x, t = create_mini_batch(positions_train_shuffled, batch_pos, args.batchsize)
        hist = v_net.fit(x, t, batch_size=args.batchsize, epochs=1, verbose=0)

        train_itr += 1
        interval_loss_sum += hist.history['loss'][0]
        interval_acc_sum += hist.history['acc'][0]
        train_loss_sum += hist.history['loss'][0]
        train_acc_sum += hist.history['acc'][0]

        # print train loss and accuracy
        if train_itr % args.eval_interval == 0:
            logging.info(
                'epoch = %s, iteration = %s, interval loss = %s, interval accuracy = %s',
                e + 1, train_itr, "{:.4f}".format(
                    interval_loss_sum / args.eval_interval), "{:.4f}".format(
                        interval_acc_sum / args.eval_interval))
            interval_loss_sum = 0
            interval_acc_sum = 0

    end_pos = batch_pos + args.batchsize
    remain_size = train_size - end_pos
    if remain_size > 0:
        x, t = create_mini_batch(positions_train_shuffled, end_pos, remain_size)
        hist = v_net.fit(x, t, batch_size=remain_size, epochs=1, verbose=0)
        train_itr += 1
        train_loss_sum += hist.history['loss'][0]
        train_acc_sum += hist.history['acc'][0]

    logging.info(
        "train finished: epoch = %s, iteration = %s, train loss = %s, train accuracy = %s",
        e + 1, train_itr, train_loss_sum / train_itr, train_acc_sum / train_itr)

    # validate
    logging.info('start validation')
    test_itr = 0
    test_loss_sum = 0
    test_acc_sum = 0
    for i in tqdm(range(0, len(positions_test) - args.batchsize, args.batchsize)):
        x, t = create_mini_batch(positions_test, i, args.batchsize)
        [loss, acc] = v_net.evaluate(x, t, verbose=0)
        test_itr += 1
        test_loss_sum += loss
        test_acc_sum += acc
    logging.info(
        'validation finished: epoch = %s, iteration = %s, test loss = %s, test accuracy = %s',
        e + 1, test_itr, test_loss_sum / test_itr, test_acc_sum / test_itr)

    logging.info("epoch %s finished", e + 1)
    logging.info('save the model')
    v_net.save_weights('init_policy_bn_epoch{}.h5'.format(e + 1))

    # update learning rate
    v_net.optimizer.lr = v_net.optimizer.lr * 0.92
