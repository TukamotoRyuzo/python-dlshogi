import argparse
import logging
import math
import os
import pickle
import re

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical

import numpy as np

import pydlshogi.common as cmn
import pydlshogi.features as fts
from pydlshogi.network.policy_bn import PolicyNetwork
from pydlshogi.read_kifu import read_kifu

parser = argparse.ArgumentParser()
# yapf: disable
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_policy', help='model file name')
parser.add_argument('--state', type=str, default='model/state_policy', help='state file name')
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

p_net = PolicyNetwork()
p_net.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])
p_net.summary()

# Init/Resume
if args.initmodel:
    logging.info('Load model from %s', args.initmodel)
    p_net.load_weights(args.initmodel)

logging.info('read kifu start')
# 保存済みのpickleファイルがある場合、pickleファイルを読み込む
# train data
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


class KifuSequence(Sequence):
    def __init__(self, positions, batch_size):
        self.positions = positions
        self.batch_size = batch_size
        self.num_classes = 9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM

    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size > len(self.positions):
            batch_size = len(self.positions) - self.batch_size * idx
        else:
            batch_size = self.batch_size
        mini_batch_data = [None] * batch_size
        mini_batch_move = [None] * batch_size

        for b in range(batch_size):
            f, move, _ = fts.make_features(
                self.positions[idx * self.batch_size + b])
            mini_batch_data[b] = f
            mini_batch_move[b] = move
        mini_batch_move = to_categorical(
            mini_batch_move, num_classes=self.num_classes)
        return (np.array(mini_batch_data, dtype=np.float32),
                np.array(mini_batch_move, dtype=np.int32))

    def __len__(self):
        return math.ceil(len(self.positions) / self.batch_size)


# train
logging.info('start training')
train_sequence = KifuSequence(positions_train, args.batchsize)
test_sequence = KifuSequence(positions_test, args.batchsize)
p_net.fit_generator(
    train_sequence,
    steps_per_epoch=len(train_sequence),
    epochs=args.epoch,
    verbose=1,
    callbacks=[
        ModelCheckpoint(
            "policy_bn_epoch{epoch:02d}_loss{loss:.2f}_acc{acc:.2f}_valloss{val_loss:.2f}_val_acc{val_acc:.2f}.h5"
        )
    ],
    validation_data=test_sequence)
