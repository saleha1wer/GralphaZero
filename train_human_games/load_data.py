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
from utils.action_encoding import encode_action, decode_action
import pickle
import copy
import asyncio

async def analyze(boards,time_limit=0.01):
    transport, engine = await chess.engine.popen_uci(r'/opt/homebrew/opt/stockfish/bin/stockfish')
    scores = []
    best_moves = []
    for board in boards:
        info = await engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info['score'].white().score(mate_score=10000)
        move = info['pv'][0]
        best_moves.append(move)
        if score in [10000,-10000]:
            score = 1 if score > 0 else -1
        else:
            adjusted_score = score/1000
            score = min(adjusted_score,1) if score > 0 else max(adjusted_score,-1)
        scores.append(score)

    await engine.quit()
    return scores,best_moves

def get_stockfish_value_moves(boards,time_limit=0.01):
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    scores,best_moves = asyncio.run(analyze(boards,time_limit=time_limit))
    return scores,best_moves

def encode_game(pgn_string,result,stockfish_value,time_limit=0.1):
    boards, values, policies = [], [], []
    pgn = io.StringIO(pgn_string)
    game = PGN.read_game(pgn)
    board = chess.Board()
    res_to_val = {'1-0': 1, '0-1': -1, '½-½': 0}
    value = res_to_val[result]
    for move in game.mainline_moves():
        temp_board = copy.deepcopy(board)
        boards.append(temp_board)
        values.append(value)
        policy = np.zeros((8,8,73))
        act_idx = encode_action(temp_board,move)
        val = 0.5 if stockfish_value else 1
        policy[act_idx[0],act_idx[1],act_idx[2]] = val
        policies.append(policy)
        board.push(move)
    if stockfish_value:
        values, best_moves = get_stockfish_value_moves(boards,time_limit=time_limit)
        for idx, policy in enumerate(policies):
            new_idx = list(encode_action(boards[idx],best_moves[idx]))
            best_idx = list(np.unravel_index(policy.argmax(), policy.shape))
            val = 1 if new_idx == best_idx else 0.5
            policy[new_idx[0],new_idx[1],new_idx[2]] = val
    assert len(boards) == len(values) == len(policies), 'Lengths of boards, values, and policies must be equal'
    return boards, values, policies


def load_data(path,stockfish_value=True,testing=False,testing_range=(0,1000),save_path=None,time_limit=0.1):
    games_df = pd.read_csv(path)
    games_df = games_df[~games_df['result'].isna()]
    buffer = Buffer(max_size=np.inf)
    boards, values, policies = [], [], []
    df =   games_df.iloc[testing_range[0]:testing_range[1]] if testing else games_df
    for idx, row in tqdm(df.iterrows()):
        pgn_string = row['moves']
        result = row['result']
        board, value, policy  = encode_game(pgn_string,result,stockfish_value,time_limit=time_limit)
        boards.extend(board)
        values.extend(value)
        policies.extend(policy)
    buffer.push(boards, values, policies)
    if save_path is not None:
        buffer.save(save_path)
    return buffer


if __name__ == '__main__':
    # Testing
    testing_range = (0,5000)
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    buf = load_data(path=data_path,
                    testing=True,
                    stockfish_value=True,
                    testing_range=testing_range,
                    save_path='{}_{}_games_buffer'.format(testing_range[0],testing_range[1]),
                    time_limit=0.025)
