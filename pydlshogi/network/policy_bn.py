from keras.layers import Activation, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Reshape
from keras.models import Sequential

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192


class PolicyNetwork(Sequential):
    def __init__(self):
        super().__init__()
        # yapf: disable
        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal', input_shape=(104, 9, 9)))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))

        self.add(Conv2D(ch, 3, padding='same', data_format="channels_first", kernel_initializer='he_normal'))
        self.add(BatchNormalization(axis=1))
        self.add(Activation('relu'))
        
        self.add(Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, padding='same', data_format="channels_first"))
        self.add(Reshape((MOVE_DIRECTION_LABEL_NUM * 9 * 9,)))
        self.add(Activation('softmax'))
        # self.add(GlobalAveragePooling2D(data_format="channels_first"))
        # self.add(Dense(MOVE_DIRECTION_LABEL_NUM * 9 * 9))
        # self.add(Activation('softmax'))
