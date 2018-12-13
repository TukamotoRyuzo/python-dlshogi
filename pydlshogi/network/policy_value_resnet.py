from keras import layers
from keras.models import Model
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192
fcl = 256
num_classes = MOVE_DIRECTION_LABEL_NUM * 9 * 9

def _conv2D(x):
    return layers.Conv2D(
        ch,
        3,
        padding='same',
        kernel_initializer='he_normal')(x)

# resnet一つを表すブロック
# x: ------------------------------->ReLU
# y: Conv->Batch->ReLU->Conv->Batch->ReLU
def _resnet_block(x, block_id):
    y = _conv2D(x)
    y = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
    y = layers.ReLU(name='conv_{}_relu'.format(block_id))(x)
    y = _conv2D(x)
    y = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
    y = layers.add([x, y])
    y = layers.ReLU(name='conv_{}_relu'.format(block_id))(y) 
    return y

def MyPolicyValueResnet(blocks):
    # input
    board_image = layers.Input(shape=(9, 9, 104))

    # 共通。ResnetのResnetたるところ
    x = _conv2D(board_image)
    for i in range(blocks):
        x = _resnet_block(x, i)
    common_out = x

    # policy network
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        name='policy_conv_out')(common_out)
    x = layers.Reshape((num_classes,), name='policy_reshape')(x)
    policy_out = layers.Activation('softmax', name='policy_out')(x)

    # value network
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='value_conv_out')(common_out)
    x = layers.BatchNormalization(name='conv')(x)
    x = layers.Reshape((num_classes,), name='value_reshape')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(fcl, activation='relu', name='value_dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, name='value_dense_2')(x)
    value_out = layers.Activation('sigmoid', name='value_out')(x)

    policy_value_model = Model(inputs=board_image, outputs=[policy_out, value_out])

    return policy_value_model

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


