import argparse
import logging

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, training
from chainer.training import extensions

from pydlshogi.network.policy_bn import PolicyNetwork

from train_tools import PositionDataset, load_kifu_data

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

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    filename=args.log,
    level=logging.DEBUG)

# data set
logging.info("prepare data set")
positions_train, positions_test = load_kifu_data(args)
train = PositionDataset(positions_train)
test = PositionDataset(positions_test)

# iterator
train_iter = iterators.MultithreadIterator(train, args.batchsize)
test_iter = iterators.MultithreadIterator(
    test, args.batchsize, repeat=False, shuffle=False)

# model
p_net = PolicyNetwork()
model = L.Classifier(p_net, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
print(sum(p.data.size for p in model.params()))

# optimizer
opt = chainer.optimizers.SGD(lr=args.lr).setup(model)

# updater
updater = training.StandardUpdater(train_iter, opt, device=0)
trainer = training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out='result')

# extensions
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.LogReport(log_name=args.log))

if extensions.PlotReport.available():
    # trainer.extend(
    #     extensions.PlotReport(
    #         ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    # trainer.extend(
    #     extensions.PlotReport(
    #         ['main/accuracy', 'validation/main/accuracy'],
    #         'epoch',
    #         file_name='accuracy.png'))
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
            'validation/main/accuracy', 'elapsed_time'
        ]))
    trainer.extend(extensions.ProgressBar(update_interval=10))

# train
trainer.run()
