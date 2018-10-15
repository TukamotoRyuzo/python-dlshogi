from pydlshogi.usi.usi import usi
from pydlshogi.player.policy_player import PolicyPlayer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'modelfile', type=str, help='saved model weight file location')
args = parser.parse_args()

player = PolicyPlayer(args.modelfile)
usi(player)
