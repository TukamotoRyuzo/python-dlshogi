from keras import layers
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

_ch = 192
_num_classes = MOVE_DIRECTION_LABEL_NUM * 9 * 9


def PolicyNetwork():
    # input
    board_image = layers.Input(shape=(9, 9, 104))

    # convolution
    x = _conv_block(board_image, 1)
    x = _conv_block(x, 2)
    x = _conv_block(x, 3)
    x = _conv_block(x, 4)
    x = _conv_block(x, 5)
    x = _conv_block(x, 6)
    x = _conv_block(x, 7)
    x = _conv_block(x, 8)
    x = _conv_block(x, 9)
    x = _conv_block(x, 10)
    x = _conv_block(x, 11)
    x = _conv_block(x, 12)
    x = _conv_block(x, 13)

    # output
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        name='conv_out')(x)
    x = layers.Reshape((_num_classes,), name='reshape')(x)
    movement_probability = layers.Activation('softmax', name='output')(x)

    policy_model = Model(inputs=board_image, outputs=movement_probability)

    return policy_model


def _conv_block(x, block_id):
    x = layers.Conv2D(
        _ch,
        3,
        padding='same',
        kernel_initializer='he_normal',
        name='conv_{}'.format(block_id))(x)
    x = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
    x = layers.ReLU(name='conv_{}_relu'.format(block_id))(x)
    return x
