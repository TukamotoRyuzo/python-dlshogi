import numpy as np

from keras.models import Model

import pydlshogi.features as fts
from pydlshogi.network.policy import PolicyNetwork
from pydlshogi.player.base_player import BasePlayer


def greedy(logits):
    return np.argmax(logits)


def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)


class PolicyPlayer(BasePlayer):

    def __init__(self, modelfile):
        super().__init__()
        self.modelfile = modelfile
        self.model = None

    def usi(self):
        print('id name policy_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        if self.model is None:
            self.model = PolicyNetwork()
        self.model.load_weights(self.modelfile)
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        features = fts.make_input_features_from_board(self.board)
        # self.modelは確率を返すようになっている
        # logitも欲しいのでsoftmax前の値を取り出すようにする
        layer_name = 'reshape_1'
        logit_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output)
        logits = logit_model.predict(features)
        # logitをsoftmaxしてprobabilityを計算
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # 全ての合法手について
        legal_moves = []
        legal_logits = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = fts.make_output_label(move, self.board.turn)
            # 合法手とその指し手の確率(logits)を格納
            legal_moves.append(move)
            legal_logits.append(logits[label])
            # 確率を表示
            print('info string {:5} : {:.5f}'.format(move.usi(),
                                                     probabilities[label]))

        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(legal_logits)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        # selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
