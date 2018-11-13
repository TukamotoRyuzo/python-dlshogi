from keras.layers import Activation, Input, Dense, Conv2D, Reshape
from keras.models import Model

from pydlshogi.common import MOVE_DIRECTION_LABEL_NUM

ch = 192
fcl = 256


class PolicyValueNetwork(Model):

    def __init__(self):
        # yapf: disable
        inputs = Input((104, 9, 9))
        h1 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(inputs)
        h2 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h1)
        h3 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h2)
        h4 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h3)
        h5 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h4)
        h6 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h5)
        h7 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h6)
        h8 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h7)
        h9 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h8)
        h10 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h9)
        h11 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h10)
        h12 = Conv2D(ch, 3, activation='relu', padding='same', data_format="channels_first")(h11)

        # policy network
        h13_p = Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, activation='relu', padding='same', data_format="channels_first")(h12)
        h13_p = Reshape((MOVE_DIRECTION_LABEL_NUM * 9 * 9,))(h13_p)
        out_policy = Activation('softmax', name='policy_output')(h13_p)

        # value network
        h13_v = Conv2D(MOVE_DIRECTION_LABEL_NUM, 1, activation='relu', padding='same', data_format="channels_first")(h12)
        h13_v = Reshape((MOVE_DIRECTION_LABEL_NUM * 9 * 9,))(h13_v)
        h14_v = Dense(fcl, activation='relu')(h13_v)
        h15_v = Dense(1)(h14_v)
        out_value = Activation('sigmoid', name='value_output')(h15_v)
        # yapf: enable

        super().__init__(inputs=inputs, outputs=[out_policy, out_value])
