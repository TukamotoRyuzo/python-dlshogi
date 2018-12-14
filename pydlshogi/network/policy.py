import chainer
import chainer.links as L
import chainer.functions as F

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

_ch = 192


class PolicyNetwork(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.init_conv = _PNBlock(104)
            self.conv_blocks = _PNBlock(_ch).repeat(12)
            self.out_conv = _OutConv()

    @profile
    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv_blocks(x)
        x = self.out_conv(x)
        if chainer.config.train:
            return x
        return F.softmax(x)


class _PNBlock(chainer.Chain):

    def __init__(self, in_channels):
        super().__init__()
        he_w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, _ch, ksize=3, pad=1, nobias=True, initialW=he_w)

    @profile
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class _OutConv(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.out_conv = L.Convolution2D(_ch, MOVE_DIRECTION_LABEL_NUM, ksize=1, nobias=True)

    @profile
    def forward(self, x):
        x = self.out_conv(x)
        x = F.reshape(x, (-1, 9 * 9 * MOVE_DIRECTION_LABEL_NUM))
        return x
