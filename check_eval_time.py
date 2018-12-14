import argparse

import chainer

from chainer import iterators, serializers

# from pydlshogi.network.shufflenet_policy import ShufflePolicy
from pydlshogi.network.policy_bn import PolicyNetwork

from train_tools import PositionDataset, load_kifu_data

from tqdm import tqdm

import numpy as np
import time

parser = argparse.ArgumentParser()
# yapf: disable
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoch times')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
# yapf: enable
args = parser.parse_args()

chainer.config.train = False
chainer.config.enable_backprop = False

# model = ShufflePolicy()
model = PolicyNetwork()
model.to_gpu(0)
# serializers.load_npz('./model/model_policy_shuffle_chainer.model', model)
# serializers.load_npz('./model/model_policy_bn_chainer.model', model)

positions_train, positions_test = load_kifu_data(args)
test = PositionDataset(positions_test)
test_iter = iterators.MultithreadIterator(
    test, args.batchsize, repeat=False, shuffle=False)

total = 0
for x in tqdm(test_iter):
    x = np.array([d[0] for d in x])
    x = model.xp.asarray(x)
    start = time.time()
    y = model(x)
    total += time.time() - start

print(total)
