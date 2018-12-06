import chainer
import chainer.links as L
import chainer.functions as F

_ch = 192


class PolicyNetwork(chainer.ChainList):

    def __init__(self):
        super().__init__(
            _PNBlock(104), _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch),
            _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch),
            _PNBlock(_ch), _PNBlock(_ch), _PNBlock(_ch))

    def forward(self, x):
        for f in self.children():
            x = f(x)
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
            self.bn = L.BatchNormalization(_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
