from keras import layers
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192
fcl = 256

num_classes = MOVE_DIRECTION_LABEL_NUM * 9 * 9


def PolicyValueNetwork():
    # input
    board_image = layers.Input(shape=(9, 9, 104))

    # common part
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
    common_out = _conv_block(x, 13)

    # policy part
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        name='policy_conv_out')(common_out)
    x = layers.Reshape((num_classes,), name='policy_reshape')(x)
    policy_out = layers.Activation('softmax', name='policy_out')(x)

    # value part
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='value_conv_out')(common_out)
    x = layers.Reshape((num_classes,), name='value_reshape')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(fcl, activation='relu', name='value_dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, name='value_dense_2')(x)
    value_out = layers.Activation('sigmoid', name='value_out')(x)

    policy_value_model = Model(inputs=board_image, outputs=[policy_out, value_out])

    return policy_value_model


def _conv_block(x, block_id):
    x = layers.Conv2D(
        ch,
        3,
        padding='same',
        kernel_initializer='he_normal',
        name='conv_{}'.format(block_id))(x)
    x = layers.BatchNormalization(name='conv_{}_bn'.format(block_id))(x)
    x = layers.ReLU(name='conv_{}_relu'.format(block_id))(x)
    return x
