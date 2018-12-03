import keras.backend as K
from keras import layers
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

_ch = 192
_classes = 9 * 9 * MOVE_DIRECTION_LABEL_NUM


def ShufflePolicy():
    board_image = layers.Input(shape=(9, 9, 104))

    # Initial Convolution
    x = layers.Conv2D(
        _ch,
        3,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False,
        name='initial_convolution')(board_image)
    x = layers.BatchNormalization(name='initinal_bn')(x)
    x = layers.ReLU(6., name='initial_relu')(x)

    # shuffle convolution
    x = _shufflenet_unit(x, 4, 1)
    x = _shufflenet_unit(x, 4, 2)
    x = _shufflenet_unit(x, 4, 3)
    x = _shufflenet_unit(x, 4, 4)
    x = _shufflenet_unit(x, 4, 5)
    x = _shufflenet_unit(x, 4, 6)
    x = _shufflenet_unit(x, 4, 7)
    x = _shufflenet_unit(x, 4, 8)
    x = _shufflenet_unit(x, 4, 9)
    x = _shufflenet_unit(x, 4, 10)
    x = _shufflenet_unit(x, 4, 11)
    x = _shufflenet_unit(x, 4, 12)
    x = _shufflenet_unit(x, 4, 13)

    # output
    # x = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    # move_probs = layers.Dense(_classes, activation='softmax')(x)
    x = layers.Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, padding='same', name='conv_out')(x)
    x = layers.Reshape((_classes,), name='reshape')(x)
    move_probs = layers.Activation('softmax', name='output')(x)

    shuffle_policy_model = Model(inputs=board_image, outputs=move_probs)

    return shuffle_policy_model


def _shufflenet_unit(inputs, g_num, block_id):

    channels = inputs.shape.as_list()[-1]
    assert channels % g_num == 0, 'group number {} must be a divisor of channels number {}'.format(
        g_num, channels)

    # group conv
    x = _group_conv(inputs, g_num, block_id, 'first')
    x = layers.BatchNormalization(name='shuffle_{}_{}_bn'.format(block_id, 'first'))(x)
    x = layers.ReLU(6., name='shuffle_{}_relu'.format(block_id))(x)

    # shuffle channels
    x = layers.Lambda(_channel_shuffle, arguments={'g_num': g_num})(x)

    # depthwise conv
    x = layers.DepthwiseConv2D(
        3,
        padding='same',
        use_bias=False, 
        depthwise_initializer='he_normal',
        name='conv_dw_{}'.format(block_id))(x)
    x = layers.BatchNormalization(name='conv_dw_{}_bn'.format(block_id))(x)

    # group conv
    x = _group_conv(x, g_num, block_id, 'last')
    x = layers.BatchNormalization(name='shuffle_{}_{}_bn'.format(block_id, 'last'))(x)

    # output
    x = layers.add([inputs, x], name='shuffle_{}_out_add'.format(block_id))
    x = layers.ReLU(6., name='shuffle_{}_out_relu'.format(block_id))(x)

    return x


def _group_conv(x, g_num, block_id, position):
    channels = x.shape.as_list()[-1]
    g_size = channels // g_num
    out_list = []
    for g in range(g_num):
        offset = g * g_size
        group = layers.Lambda(
            lambda z, ofs=offset: z[:, :, :, ofs:ofs + g_size],
            name='gconv_{}_{}_{}_slice'.format(block_id, position, g + 1))(x)
        group = layers.Conv2D(
            g_size,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='gconv_{}_{}_{}'.format(block_id, position, g + 1))(group)
        out_list.append(group)
    out = layers.Concatenate(
        name='gconv_{}_{}_concat'.format(block_id, position))(out_list)
    return out


def _channel_shuffle(x, g_num):
    height, width, channels = x.shape.as_list()[1:]
    g_size = channels // g_num
    x = K.reshape(x, (-1, height, width, g_num, g_size))  # divide channels
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))          # transpose them
    x = K.reshape(x, (-1, height, width, channels))       # flatten
    return x
