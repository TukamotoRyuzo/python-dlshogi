from pydlshogi.usi.usi import usi
from pydlshogi.player.p_by_v_player import PolicyByValuePlayer
import argparse

player = PolicyByValuePlayer()
usi(player)
