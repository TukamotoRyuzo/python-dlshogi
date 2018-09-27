import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from pydlshogi.common import *

ch = 192
class PolicyNetwork():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (9, 9, 104)))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = ch, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(filters = MOVE_DIRECTION_LABEL_NUM, kernel_size = (1, 1), activation = 'softmax', padding = 'same', use_bias = True))
        self.model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(),metrics = ['accuracy'])

    def __call__(self, x):
        return self.model.predict(x)

