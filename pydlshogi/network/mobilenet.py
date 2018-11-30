from keras import layers
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

_ch = 192
_classes = 9 * 9 * MOVE_DIRECTION_LABEL_NUM


def PolicyMobileNetwork():
    board_image = layers.Input(shape=(9, 9, 104))
    # Initial Convolution
    x = layers.Conv2D(
        _ch,
        3,
        padding='same',
        kernel_initializer='he_normal',
        name='initial_convolution')(board_image)
    x = layers.BatchNormalization(name='initinal_bn')(x)
    x = layers.ReLU(6., name='initial_relu')(x)

    # Separable Convolution
    x = _separable_conv_block(x, 1)
    x = _separable_conv_block(x, 2)
    x = _separable_conv_block(x, 3)
    x = _separable_conv_block(x, 4)
    x = _separable_conv_block(x, 5)
    x = _separable_conv_block(x, 6)
    x = _separable_conv_block(x, 7)
    x = _separable_conv_block(x, 8)
    x = _separable_conv_block(x, 9)
    x = _separable_conv_block(x, 10)
    x = _separable_conv_block(x, 11)
    x = _separable_conv_block(x, 12)
    x = _separable_conv_block(x, 13)

    # output
    x = layers.Conv2D(
        MOVE_DIRECTION_LABEL_NUM,
        1,
        padding='same',
        name='conv_out')(x)
    x = layers.Reshape((_classes,), name='reshape')(x)
    movement_probability = layers.Activation('softmax', name='output')(x)

    model = Model(board_image, movement_probability, name='PoliceMobileNetwork')

    return model


def _separable_conv_block(x, block_id):
    # Depthwise Convolution
    x = layers.DepthwiseConv2D(
        3,
        padding='same',
        depthwise_initializer='he_normal',
        name='conv_dw_{}'.format(block_id))(x)
    x = layers.BatchNormalization(name='conv_dw_{}_bn'.format(block_id))(x)
    x = layers.ReLU(name='conv_dw_{}_relu'.format(block_id))(x)

    # Pointwise Convolution
    x = layers.Conv2D(
        _ch,
        1,
        padding='same',
        kernel_initializer='he_normal',
        name='conv_pw_{}'.format(block_id))(x)
    x = layers.BatchNormalization(name='conv_pw_{}_bn'.format(block_id))(x)
    x = layers.ReLU(name='conv_pw_{}_relu'.format(block_id))(x)

    return x
