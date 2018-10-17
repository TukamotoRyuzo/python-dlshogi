import argparse
import random
import pickle
import os
import re
import logging

import numpy as np

import pydlshogi.features as fts
from pydlshogi.read_kifu import read_kifu
from pydlshogi.network.value import ValueNetwork

from keras.optimizers import SGD

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
v_net.compile(SGD(lr=args.lr), 'binary_crossentropy', metrics=['accuracy'])
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
def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = fts.make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = fts.make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


# train
logging.info('start training')
itr = 0
sum_loss = 0
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train, len(positions_train))

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in tqdm(range(0, len(positions_train_shuffled) - args.batchsize, args.batchsize)):
        x, t = mini_batch(positions_train_shuffled, i, args.batchsize)
        hist = v_net.fit(x, t, batch_size=args.batchsize, epochs=1, verbose=0)
        itr += 1
        sum_loss += hist.history['loss'][0]
        itr_epoch += 1
        sum_loss_epoch += hist.history['loss'][0]
        iteration = int(i / args.batchsize)

        # print train loss and test accuracy
        if iteration % args.eval_interval == 0:
            x, t = mini_batch_for_test(positions_test, args.test_batchsize)
            y = v_net.evaluate(x, t, verbose=0)
            logging.info('epoch = %s, iteration = %s, loss = %s, accuracy = %s', e + 1,
                         iteration, sum_loss / itr, y[1])
            itr = 0
            sum_loss = 0

    # validate test data
    logging.info('validate test data')
    itr_test = 0
    sum_test_loss = 0
    sum_test_accuracy = 0
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, t = mini_batch(positions_test, i, args.batchsize)
        y = v_net.evaluate(x, t, verbose=0)
        itr_test += 1
        sum_test_loss += y[0]
        sum_test_accuracy += y[1]
    logging.info('epoch = %s, iteration = %s, test loss = %s, test accuracy = %s',
                 e + 1, iteration, sum_test_loss / itr_test,
                 sum_test_accuracy / itr_test)


logging.info('save the model')
v_net.save_weights('value_init.h5')
