from keras.layers import Activation, Conv2D, Reshape
from keras.models import Sequential
from keras.optimizers import SGD

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192


class PolicyNetwork(Sequential):

    def __init__(self):
        super().__init__()
        # yapf: disable
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first", input_shape=(104, 9, 9)))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first"))
        self.add(Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, activation='relu', padding='same', data_format="channels_first"))
        self.add(Reshape((MOVE_DIRECTION_LABEL_NUM * 9 * 9,)))
        self.add(Activation('softmax'))
        self.compile(SGD(), 'categorical_crossentropy', metrics=['accuracy'])
