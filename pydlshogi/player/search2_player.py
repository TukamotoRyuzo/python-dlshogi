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


class Search2Player(BasePlayer):
    def __init__(self):
        super().__init__()
        self.p_modelfile = r"C:\Users\NPC05027\opt\python-dlshogi\model\model_policy_bn"
        self.v_modelfile = r"C:\Users\NPC05027\opt\python-dlshogi\model\model_value_bn"
        self.p_model = None
        self.v_model = None

    def usi(self):
        print('id name search2_player')
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
        legal_win_probs = []
        legal_search_nodes = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = fts.make_output_label(move, self.board.turn)
            # 価値ネットの精度が低いので、方策ネットで着手確率が低い手は枝刈りしておく
            th = 0.05
            while not np.any(move_probs > th):
                th /= 5

            if move_probs[label] > th:
                print("first if")
                # 有望な手とその着手確率を保存
                legal_moves.append(move)
                legal_move_probs.append(move_probs[label])
                # もう1手進める
                self.board.push(move)

                # 詰んでたらそれでOK
                if self.board.is_game_over():
                    legal_win_probs.append(1)
                    continue

                # 詰んでないなら勝率を調べるため、もう1手進める

                # 枝刈りのために着手確率を用意する
                next_features = np.array(
                    [fts.make_input_features_from_board(self.board)])
                next_move_probs = self.p_model.predict(next_features).ravel()

                # 次の合法手から盤面を生成する
                next_legal_feats = []
                search_nodes = 0
                for next_move in self.board.legal_moves:
                    search_nodes += 1
                    print("second for")
                    next_label = fts.make_output_label(next_move,
                                                       self.board.turn)
                    th = 0.05
                    while not np.any(next_move_probs > th):
                        th /= 5
                    if next_move_probs[next_label] > 0.05:
                        print("second if")
                        self.board.push(next_move)
                        next_legal_feats.append(
                            fts.make_input_features_from_board(self.board))
                        self.board.pop()
                # 勝率を計算する
                print("calc win rate")
                x = np.array(next_legal_feats, dtype=np.float)
                next_legal_win_probs = self.v_model.predict_on_batch(x).ravel()
                # 最小勝率を採用する
                legal_win_probs.append(min(next_legal_win_probs))
                legal_search_nodes.append(search_nodes)
                print("go next")
                self.board.pop()

        prob_idx = np.argsort(legal_win_probs)
        for idx in prob_idx:
            wp = legal_win_probs[idx]
            move = legal_moves[idx]
            if wp == 1.0:
                cp = 30000
            else:
                cp = int(-math.log(1.0 / wp - 1.0) * 600)
            print('info depth {} nodes {} score cp {} pv {}'.format(
                2, legal_search_nodes[idx], cp, move.usi()))

        # 確率が最大の手を選ぶ(グリーディー戦略)
        print(legal_win_probs)
        selected_index = greedy(legal_win_probs)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        # selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
