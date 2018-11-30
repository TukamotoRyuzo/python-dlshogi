from keras.layers import Activation, Conv2D, Reshape
from keras.models import Sequential

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192


class PolicyNetwork(Sequential):

    def __init__(self):
        super().__init__()
        # yapf: disable
        self.add(Conv2D(ch, 3, activation='relu', padding='same', input_shape=(9, 9, 104)))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(ch, 3, activation='relu', padding='same'))
        self.add(Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, activation='relu', padding='same'))
        self.add(Reshape((MOVE_DIRECTION_LABEL_NUM * 9 * 9,)))
        self.add(Activation('softmax'))
