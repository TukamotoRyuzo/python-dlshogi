#from keras import layers
#from keras.models import Model
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM
#from policy_bn import _conv_block

ch = 192
fcl = 256

# def _resnet_block(x, block_id)
#     y = layers.Conv2D(
#         ch,
#         3,
#         padding='same',
#         kernel_initializer='he_normal',
#         name='conv_{}'.format(block_id))(x)
#     y = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
#     y = layers.ReLU(name='conv_{}_relu'.format(block_id))(x)
#     y = layers.Conv2D(
#         ch,
#         3,
#         padding='same',
#         kernel_initializer='he_normal',
#         name='conv_{}'.format(block_id))(x)
#     y = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
#     y = layers.ReLU(name='conv_{}_relu'.format(block_id))(x + y)
    
#     return y

# class MyPolicyValueResnet():
#     # input
#     board_image = layers.Input(shape=(9, 9, 104))

#     # convolution
#     x = _conv_block(board_image, 1)
#     x = _conv_block(x, 2)
#     x = _conv_block(x, 3)
#     x = _conv_block(x, 4)


class Block(Chain):

    def __init__(self):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1, nobias=True)
            self.bn2 = L.BatchNormalization(ch)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = self.bn2(self.conv2(h1))
        return F.relu(x + h2)

class PolicyValueResnet(Chain):
    def __init__(self, blocks = 5):
        super(PolicyValueResnet, self).__init__()
        self.blocks = blocks
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            for i in range(1, blocks):
                self.add_link('b{}'.format(i), Block())
            # policy network
            self.policy=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.policy_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            # value network
            self.value1=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.value1_bn = L.BatchNormalization(MOVE_DIRECTION_LABEL_NUM)
            self.value2=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.value3=L.Linear(fcl, 1)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        for i in range(1, self.blocks):
            h = self['b{}'.format(i)](h)
        # policy network
        h_policy = self.policy(h)
        u_policy = self.policy_bias(F.reshape(h_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        h_value = F.relu(self.value1_bn(self.value1(h)))
        h_value = F.relu(self.value2(h_value))
        u_value = self.value3(h_value)
        return u_policy, u_value


