import logging
import os
import pickle
import re

from keras.utils import to_categorical

import numpy as np

import pydlshogi.common as cmn
import pydlshogi.features as fts
from pydlshogi.read_kifu import read_kifu

num_classes = 9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM


def create_mini_batch(positions, batch_head, batch_size):
    mini_batch_data = np.empty((batch_size, 9, 9, 104), dtype=np.float)
    mini_batch_move = np.empty((batch_size), dtype=np.int)
    mini_batch_win = np.empty((batch_size), dtype=np.int)
    for b in range(batch_size):
        features, move, win = fts.make_features(positions[batch_head + b])
        features = np.array(features).transpose((0, 2, 3, 1))
        mini_batch_data[b] = features
        mini_batch_move[b] = move
        mini_batch_win[b] = win
    mini_batch_move = to_categorical(mini_batch_move, num_classes=num_classes)

    return (mini_batch_data, mini_batch_move, mini_batch_win.reshape((-1, 1)))


def load_kifu_data(args):
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

    return positions_train, positions_test


class Status:

    def __init__(self):
        self.policy_loss_sum = 0
        self.policy_acc_sum = 0
        self.value_loss_sum = 0
        self.value_acc_sum = 0
        self.total_loss_sum = 0

        self.iter_num = 0

        self.policy_loss = 0
        self.policy_acc = 0
        self.value_loss = 0
        self.value_acc = 0
        self.total_loss = 0

    def update(self, results):
        self.total_loss_sum += results[0]
        self.policy_loss_sum += results[1]
        self.value_loss_sum += results[2]
        self.policy_acc_sum += results[3]
        self.value_acc_sum += results[4]
        self.iter_num += 1

    def calc_mean(self):
        self.policy_loss = self.policy_loss_sum / self.iter_num
        self.policy_acc = self.policy_acc_sum / self.iter_num
        self.value_loss = self.value_loss_sum / self.iter_num
        self.value_acc = self.value_acc_sum / self.iter_num
        self.total_loss = self.total_loss_sum / self.iter_num

    def reset(self):
        self.policy_loss_sum = 0
        self.policy_acc_sum = 0
        self.value_loss_sum = 0
        self.value_acc_sum = 0
        self.total_loss_sum = 0

        self.iter_num = 0
