import argparse
import logging
import os
import pickle
import random
import re

from keras.metrics import binary_accuracy, categorical_accuracy
from keras.optimizers import SGD

from tqdm import tqdm

import numpy as np

import pydlshogi.common as cmn
import pydlshogi.features as fts
from pydlshogi.network.policy_value import PolicyValueNetwork
from pydlshogi.read_kifu import read_kifu

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
opts = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
pv_net.compile(
    optimizer=SGD(lr=args.lr),
    loss={
        'policy_output': 'categorical_crossentropy',
        'value_output': 'binary_crossentropy'
    },
    loss_weights={
        'policy_output': 1,
        'value_output': 1
    },
    metrics=['accuracy'])
pv_net.summary()

# Init/Resume
if args.initmodel:
    logging.info('Load model from %s', args.initmodel)
    pv_net.load_weights(args.initmodel)

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
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = fts.make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)
    mini_batch_move = np.identity(
        9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM)[mini_batch_move]

    return (np.array(mini_batch_data, dtype=np.float32),
            np.array(mini_batch_move, dtype=np.int32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = fts.make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    mini_batch_move = np.identity(
        9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM)[mini_batch_move]
    return (np.array(mini_batch_data, dtype=np.float32),
            np.array(mini_batch_move, dtype=np.int32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


# train
logging.info('start training')
itr = 0
sum_loss = 0
sum_acc_p = 0
sum_acc_v = 0
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train,
                                             len(positions_train))

    for i in tqdm(
            range(0,
                  len(positions_train_shuffled) - args.batchsize,
                  args.batchsize)):
        x, t1, t2 = mini_batch(positions_train_shuffled, i, args.batchsize)
        hist = pv_net.fit(
            x, {
                'policy_output': t1,
                'value_output': t2
            },
            batch_size=args.batchsize,
            epochs=1,
            verbose=0)

        itr += 1
        sum_loss += hist.history['loss'][0]
        sum_acc_p += hist.history['policy_output_acc'][0]
        sum_acc_v += hist.history['value_output_acc'][0]

        iteration = int(i / args.batchsize)

        # print train loss and accuracy
        if iteration % args.eval_interval == 0:
            logging.info(
                'epoch = %s, iteration = %s, train loss = %s, train accuracy = %s, %s',
                e + 1, iteration, sum_loss / itr, sum_acc_p / itr, sum_acc_v / itr)
            itr = 0
            sum_loss = 0
            sum_acc_p = 0
            sum_acc_v = 0

    # validate test data
    logging.info('validate test data')
    itr_test = 0
    sum_train_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, t1, t2 = mini_batch(positions_test, i, args.batchsize)
        [loss, policy_loss, value_loss, policy_acc,
         value_acc] = pv_net.evaluate(
             x, {
                 'policy_output': t1,
                 'value_output': t2
             }, verbose=0)
        itr_test += 1
        sum_train_loss += loss
        sum_test_accuracy1 += policy_acc
        sum_test_accuracy2 += value_acc
    logging.info(
        'epoch = %s, iteration = %s, test loss = %s, test accuracy = %s, %s',
        e + 1, iteration, sum_train_loss / itr_test,
        sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test)

    logging.info('save the model')
    if args.initmodel:
        pv_net.save_weights(args.initmodel)
    else:
        pv_net.save_weights('init_valuepolicy_epoch{}.h5'.format(e + 1))
