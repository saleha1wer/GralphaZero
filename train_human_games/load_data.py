"""
Data Gathering
    - Gather data of stockfish vs stockfish
    - Gather human data
"""
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import pandas as pd
from buffer import Buffer
from tqdm import tqdm
import io 
import numpy as np
import chess.pgn as PGN
import chess
from utils.action_encoding import encode_action

def encode_game(pgn_string,result):
    boards, values, policies = [], [], []
    pgn = io.StringIO(pgn_string)
    game = PGN.read_game(pgn)
    board = chess.Board()
    res_to_val = {'1-0': 1, '0-1': -1, '½-½': 0}
    value = res_to_val[result]
    for move in game.mainline_moves():
        boards.append(board)
        turn_multiplier = 1 if board.turn else -1
        values.append(value*turn_multiplier)
        policy = np.zeros((8,8,73))
        act_idx = encode_action(board,move)
        policy[act_idx[0],act_idx[1],act_idx[2]] = 1
        policies.append(policy)
        board.push(move)
    assert len(boards) == len(values) == len(policies), 'Lengths of boards, values, and policies must be equal'
    return boards, values, policies


def load_data(path,testing=False):
    games_df = pd.read_csv(path)
    games_df = games_df[~games_df['result'].isna()]
    buffer = Buffer(max_size=np.inf)
    if testing:

        for idx,row in tqdm(games_df.iloc[:1000].iterrows()):
            pgn_string = row['moves']
            result = row['result']
            boards, values, policies  = encode_game(pgn_string,result)
            buffer.push(boards, values, policies)
        return buffer
    else:
        for idx, row in tqdm(games_df.iterrows()):
            pgn_string = row['moves']
            result = row['result']
            boards, values, policies  = encode_game(pgn_string,result)
            buffer.push(boards, values, policies)
        return buffer


if __name__ == '__main__':
    # Testing
    # buf = load_data('/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv',testing=True)
    # print(buf.__len__())
    # print(buf.buffer['value'])
    # print(buf.sample_all())
    pass