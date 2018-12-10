import logging
import os
import pickle
import re

import numpy as np

import pydlshogi.common as cmn
import pydlshogi.features as fts
from pydlshogi.read_kifu import read_kifu

from tqdm import tqdm

num_classes = np.int32(9 * 9 * cmn.MOVE_DIRECTION_LABEL_NUM)


class PositionDataset:

    def __init__(self, positions):
        self._positions = positions
        self._length = len(positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop
            batch_length = stop - start
            feature = np.empty((batch_length, 104, 9, 9), dtype=np.float32)
            move = np.empty((batch_length), dtype=np.int32)
            win = np.empty((batch_length), dtype=np.int32)
            for i in range(batch_length):
                f, m, w = fts.make_features(self._positions[start + i])
                feature[i] = f
                move[i] = m
                win[i] = w
        else:
            feature, move, win = fts.make_features(self._positions[index])
            feature = np.array(feature, dtype=np.float32)
            move = np.int32(move)
            win = np.int32(win)
            
        return (feature, move)

    def __len__(self):
        return self._length


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
