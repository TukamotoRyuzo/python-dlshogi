import math

import numpy as np

import pydlshogi.features as fts
from pydlshogi.network.policy_bn import PolicyNetwork
from pydlshogi.network.value_bn import ValueNetwork
from pydlshogi.player.base_player import BasePlayer


def greedy(logits):
    return np.argmax(logits)


def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)


class PolicyByValuePlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.p_modelfile = r"C:\Users\NPC05027\opt\python-dlshogi\model\model_policy_bn"
        self.v_modelfile = r"C:\Users\NPC05027\opt\python-dlshogi\model\model_value_bn"
        self.p_model = None
        self.v_model = None

    def usi(self):
        print('id name p_by_v_player')
        print('option name p_modelfile type string default ' +
              self.p_modelfile)
        print('option name v_modelfile type string default ' +
              self.v_modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'p_modelfile':
            self.p_modelfile = option[3]
        elif option[1] == 'v_modelfile':
            self.v_modelfile = option[3]

    def isready(self):
        if self.p_model is None:
            self.p_model = PolicyNetwork()
        self.p_model.load_weights(self.p_modelfile)
        if self.v_model is None:
            self.v_model = ValueNetwork()
        self.v_model.load_weights(self.v_modelfile)
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        features = np.array([fts.make_input_features_from_board(self.board)])
        move_probs = self.p_model.predict(features).ravel()

        # 全ての合法手について
        legal_moves = []
        legal_move_probs = []
        legal_feats = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = fts.make_output_label(move, self.board.turn)
            # 価値ネットの精度が低いので、方策ネットで着手確率5%未満の手は枝刈りしておく
            if move_probs[label] > 0.05:
                # 合法手とその指し手の確率(logits)を格納
                legal_moves.append(move)
                legal_move_probs.append(move_probs[label])
                self.board.push(move)
                legal_feats.append(fts.make_input_features_from_board(self.board))
                self.board.pop()
        # 合法手の勝率を計算
        x = np.array(legal_feats, dtype=np.float)
        legal_win_probs = 1 - self.v_model.predict_on_batch(x).ravel()

        total_probs = np.array(legal_move_probs) * np.array(legal_win_probs)
        prob_idx = np.argsort(total_probs)
        for idx in prob_idx:
            wp = legal_win_probs[idx]
            move = legal_moves[idx]
            if wp == 1.0:
                cp = 30000
            else:
                cp = int(-math.log(1.0 / wp - 1.0) * 600)
            print('info depth {} score cp {} pv {}'.format(int(total_probs[idx]*100), cp, move.usi()))

        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(total_probs)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        # selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
