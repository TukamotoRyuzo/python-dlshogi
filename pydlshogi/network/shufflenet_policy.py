import chainer
import chainer.links as L
import chainer.functions as F

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

_ch = 192
_classes = 9 * 9 * MOVE_DIRECTION_LABEL_NUM


class ShufflePolicy(chainer.Chain):

    def __init__(self):
        super().__init__()
        he_w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.init_conv = L.Convolution2D(
                104, _ch, ksize=3, pad=1, nobias=True, initialW=he_w)
            self.init_bn = L.BatchNormalization(_ch)
            self.conv_blocks = _ShufflnetBlock(_ch, 4).repeat(12)
            self.out_conv = _OutConv()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = F.relu(x)
        x = self.conv_blocks(x)
        x = self.out_conv(x)
        if chainer.config.train:
            return x
        return F.softmax(x)


class _ShufflnetBlock(chainer.Chain):

    def __init__(self, in_channels, groups):
        super().__init__()
        he_w = chainer.initializers.HeNormal()
        self.groups = groups
        with self.init_scope():
            self.group_conv1 = L.Convolution2D(
                in_channels,
                _ch,
                ksize=3,
                pad=1,
                nobias=True,
                initialW=he_w,
                groups=groups)
            self.bn1 = L.BatchNormalization(_ch)

            self.dw_conv = L.DepthwiseConvolution2D(
                _ch, 1, 3, pad=1, nobias=True, initialW=he_w)

            self.group_conv2 = L.Convolution2D(
                in_channels,
                _ch,
                ksize=3,
                pad=1,
                nobias=True,
                initialW=he_w,
                groups=groups)
            self.bn2 = L.BatchNormalization(_ch)

    def _channel_shuffle(self, x):
        b, ch, h, w = x.shape
        group_size = ch // self.groups
        x = F.reshape(x, (b, self.groups, group_size, h, w))
        x = F.transpose(x, axes=(0, 2, 1, 3, 4))
        x = F.reshape(x, (b, ch, h, w))
        return x

    def forward(self, x):
        h = self.group_conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self._channel_shuffle(h)

        h = self.dw_conv(h)

        h = self.group_conv2(h)
        h = self.bn2(h)
        h = F.relu(h + x)
        return h


class _OutConv(chainer.Chain):

    def __init__(self):
        super().__init__()
        he_w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv_out = L.Convolution2D(
                _ch,
                MOVE_DIRECTION_LABEL_NUM,
                ksize=3,
                pad=1,
                nobias=True,
                initialW=he_w)

    def forward(self, x):
        x = self.conv_out(x)
        x = F.reshape(x, (-1, _classes))
        x = F.relu(x)
        return x
