from pydlshogi.usi.usi import usi
from pydlshogi.player.policy_bn_player import PolicyPlayer
import argparse

player = PolicyPlayer()
usi(player)
