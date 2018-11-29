import numpy as np

import pydlshogi.features as fts
from pydlshogi.network.policy import PolicyNetwork
from pydlshogi.network.value import ValueNetwork
from pydlshogi.player.base_player import BasePlayer

import math


def greedy(logits):
    return np.argmax(logits)


def boltzmann(prob, temperature):
    prob /= temperature
    prob = np.exp(prob)
    prob /= prob.sum()
    return np.random.choice(len(prob), p=prob)


def boltzmann2(prob, temperature):
    prob /= temperature
    prob = np.exp(prob)
    prob /= prob.sum()
    return prob


class PolicyValueSearch1Player(BasePlayer):
    def __init__(self):
        super().__init__()
        self.pnet_file = "/home/n-kobayashi/opt/python-dlshogi/init_epoch6.h5"
        self.vnet_file = "/home/n-kobayashi/opt/python-dlshogi/value_init_epoch6.h5"
        self.pnet_model = None
        self.vnet_model = None

    def usi(self):
        print('id name policy_value_search1_player')
        print('option name pnet_modelfile type string default ' +
              self.pnet_file)
        print('option name vnet_modelfile type string default ' +
              self.vnet_file)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'pnet_modelfile':
            self.pnet_file = option[3]
        elif option[1] == 'vnet_modelfile':
            self.vnet_file = option[3]

    def isready(self):
        if self.pnet_model is None:
            self.pnet_model = PolicyNetwork()
        if self.vnet_model is None:
            self.vnet_model = ValueNetwork()
        self.pnet_model.load_weights(self.pnet_file)
        self.vnet_model.load_weights(self.vnet_file)
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        features = np.array([fts.make_input_features_from_board(self.board)])
        probabilities = self.pnet_model.predict(features).ravel()

        # 全ての合法手について、選択確率を求める
        legal_moves = []
        legal_probs = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = fts.make_output_label(move, self.board.turn)
            # 合法手とその指し手の確率を格納
            legal_moves.append(move)
            legal_probs.append(probabilities[label])
        # 確率の高い順に調べる
        prob_idx = np.argsort(legal_probs)[::-1]

        # 着手確率の高い順に勝率を調べる
        value_features = []
        total_prob = 0
        candidate_moves = []
        candidate_probs = []
        for idx in prob_idx:
            # 着手確率の合計が0.9を越えたらそれ以上は調べない
            if total_prob > 0.9:
                break
            total_prob += legal_probs[idx]
            candidate_probs.append(legal_probs[idx])
            move = legal_moves[idx]
            candidate_moves.append(move)
            self.board.push(move)
            value_features.append(
                fts.make_input_features_from_board(self.board))
            self.board.pop()
        x = np.array(value_features, dtype=np.float)
        value_probs = -self.vnet_model.predict(x).ravel()

        # 勝率の高い順に並べる
        value_prob_idx = np.argsort(value_probs)
        combined_probs = np.zeros(len(value_prob_idx))
        for idx in value_prob_idx:
            cmb_p = 0.6 * (1 + value_probs)[idx] + 0.4 * candidate_probs[idx]
            combined_probs[idx] = cmb_p

        comb_prob_idx = np.argsort(combined_probs)
        for idx in comb_prob_idx:
            print('info string {:5} : v{:.5f} p{:.5f} c{:.5f}'.format(
                candidate_moves[idx].usi(), value_probs[idx],
                candidate_probs[idx], combined_probs[idx]))

        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(combined_probs)

        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        # selected_index = boltzmann(combined_probs, 0.5)

        best_wp = 1 + value_probs[selected_index]
        # 勝率が0.01%切ってたら投了
        # if best_wp < 0.0001:
        #     print(best_wp)
        #     print('bestmove resign')
        #     return

        if best_wp == 1.0:
            cp = 30000
        elif best_wp == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / best_wp - 1.0) * 600)

        bestmove = candidate_moves[selected_index]

        print('info score cp {} pv {}'.format(cp, bestmove.usi()))
        print('bestmove', bestmove.usi())
