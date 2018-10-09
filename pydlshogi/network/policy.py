import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D

from pydlshogi.common import *

ch = 192
class PolicyNetwork(Sequential):
    def __init__(self):
        super().__init__()
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first", input_shape = (104, 9, 9)))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', data_format = "channels_first"))
        self.add(Conv2D(filters = MOVE_DIRECTION_LABEL_NUM, kernel_size = (1, 1), activation = 'relu', padding = 'same',
         use_bias = True, data_format = "channels_first"))
        self.add(Dence(MOVE_DIRECTION_LABEL_NUM * 9 * 9, activation='softmax'))
        self.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.SGD(), metrics = ['accuracy'])
