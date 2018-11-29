from pydlshogi.usi.usi import usi
from pydlshogi.player.policy_value_search1_player import PolicyValueSearch1Player
import argparse

player = PolicyValueSearch1Player()
usi(player)
