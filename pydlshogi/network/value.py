from keras.layers import Activation, Conv2D, Dense
from keras.models import Sequential

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192
fcl = 256


class ValueNetwork(Sequential):

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
        self.add(Dense(fcl, activation='relu'))
        self.add(Dense(1))
        self.add(Activation('sigmoid'))
