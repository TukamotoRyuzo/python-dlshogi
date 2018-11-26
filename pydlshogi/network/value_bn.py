from keras import layers
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192
fcl = 256

num_classes = MOVE_DIRECTION_LABEL_NUM * 9 * 9


def ValueNetwork():
    # input
    board_image = layers.Input(shape=(104, 9, 9))

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

    # value part
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        data_format="channels_first",
        name='conv_out')(x)
    x = layers.Reshape((num_classes,), name='reshape')(x)
    x = layers.Dense(fcl, activation='relu', name='dense_1')(x)
    x = layers.Dense(1, name='dense_2')(x)
    win_probability = layers.Activation('sigmoid', name='value_out')(x)

    value_model = Model(inputs=board_image, outputs=win_probability)

    return value_model


def _conv_block(x, block_id):
    x = layers.Conv2D(
        ch,
        3,
        padding='same',
        data_format="channels_first",
        kernel_initializer='he_normal',
        name='conv_{}'.format(block_id))(x)
    x = layers.BatchNormalization(axis=1, name='conv_{}_bn'.format(block_id))(x)
    x = layers.ReLU(name='conv_{}_relu'.format(block_id))(x)
    return x
