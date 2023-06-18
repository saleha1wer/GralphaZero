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
from utils.action_encoding import encode_action, decode_action, get_num_edges, old_decode_action, old_encode_action
import pickle
import copy
import asyncio
from scipy.special import softmax
async def analyze(boards, policy_format='graph', time_limit=0.01, return_policies=False, disable=True, move_lists=False):
    transport, engine = await chess.engine.popen_uci(r'/opt/homebrew/opt/stockfish/bin/stockfish')
    scores = []
    best_moves = []
    graph_policies = []
    array_policies = []
    for board in tqdm(boards, disable=disable):
        info = await engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info['score'].white().score(mate_score=10000)
        if score in [10000, -10000]:
            score = 1 if score > 0 else -1
        else:
            adjusted_score = score/650
            score = min(adjusted_score, 1) if score > 0 else max(adjusted_score, -1)
        score = score if board.turn == chess.WHITE else -1 * score
        scores.append(score)
        if return_policies:
            moves = info['pv']
            moves = [move for move in moves if move in list(board.legal_moves)]
            probs = [len(moves) - i for i in range(len(moves))]
            probs = softmax(probs)
            if policy_format in ['graph','both']:
                idxs = [encode_action(board, move) for move in moves]
                graph_policy = np.zeros(get_num_edges(board))
                graph_policy[idxs] = probs
                graph_policies.append(graph_policy)
            if policy_format in ['array','both']:
                array_policy = np.zeros((8, 8, 73))
                for move, prob in zip(moves, probs):
                    idx = old_encode_action(board, move)
                    array_policy[idx[0], idx[1], idx[2]] = prob
                array_policies.append(array_policy)
        else:
            if move_lists:
                move_s = info['pv']
                move = [move for move in move_s if move in list(board.legal_moves)]
            else:
                move = info['pv'][0]
            best_moves.append(move)
    await engine.quit()

    if return_policies:
        if policy_format == 'both':
            return scores, graph_policies, array_policies
        elif policy_format == 'graph':
            return scores, graph_policies
        else:
            return scores, array_policies
    else:
        return scores, best_moves


def get_stockfish_value_moves(boards, policy_format='graph', time_limit=0.01, return_policies=False, disable=True, move_lists=False):
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    return asyncio.run(analyze(boards, policy_format=policy_format, time_limit=time_limit, return_policies=return_policies, disable=disable, move_lists=move_lists))

def encode_game(pgn_string,result,stockfish_value,time_limit=0.1,disable=True,policy_format='graph'):
    boards, values, graph_policies,array_policies = [], [], [],[]
    pgn = io.StringIO(pgn_string)
    game = PGN.read_game(pgn)
    board = chess.Board()
    res_to_val = {'1-0': 1, '0-1': -1, '½-½': 0}
    value = res_to_val[result]
    for move in game.mainline_moves():
        temp_board = copy.deepcopy(board)
        boards.append(temp_board)
        eval = value if temp_board.turn else -value
        values.append(eval)
        val = np.random.uniform(0.1,0.3) if stockfish_value else 1
        if policy_format in ['graph','both']:
            act_idx = encode_action(temp_board,move)
            num_edges = get_num_edges(temp_board)
            graph_policy = np.zeros(num_edges)
            graph_policy[act_idx] = val
            graph_policies.append(graph_policy)
        if policy_format in ['array','both']:
            act_idx = old_encode_action(temp_board,move)
            array_policy = np.zeros((8,8,73))
            array_policy[act_idx[0],act_idx[1],act_idx[2]] = val
            array_policies.append(array_policy)
        board.push(move)
    if stockfish_value:
        values, best_moves = get_stockfish_value_moves(boards,time_limit=time_limit,return_policies=False,disable=disable,policy_format=policy_format)
        for idx, policy in enumerate(graph_policies):
            new_idx = encode_action(boards[idx],best_moves[idx])
            best_idx = np.argmax(policy)
            val2 = 1 if new_idx == best_idx else 1-policy[best_idx]
            policy[new_idx] = val2
        for idx, policy in enumerate(array_policies):
            new_idx = old_encode_action(boards[idx],best_moves[idx])
            best_idx = list(np.unravel_index(np.argmax(policy),policy.shape))
            val2 = 1 if new_idx == best_idx else 1-policy[best_idx[0],best_idx[1],best_idx[2]]
            policy[new_idx[0],new_idx[1],new_idx[2]] = val2
        # if policy_format == 'both':
        #     values, graph_policies, array_policies = get_stockfish_value_moves(boards, policy_format=policy_format, return_policies=True, time_limit=0.01, disable=False)
        # elif....
        #     values, graph_policies = get_stockfish_value_moves(boards, policy_format=policy_format, return_policies=True, time_limit=0.01, disable=False)
    assert len(boards) == len(values) == np.max([len(array_policies),len(graph_policies)]), 'Lengths of boards, values, and policies must be equal'
    if policy_format == 'graph':
        return boards, values, graph_policies
    elif policy_format == 'array':
        return boards, values, array_policies
    else:
        return boards, values, graph_policies, array_policies
def gen_positions(num, policy_format='graph'):
    boards = []
    pos_count = 0
    pbar = tqdm(total=num)
    while pos_count < num:
        board = chess.Board()
        move_count = 0
        n_moves = np.random.randint(20,250)
        while board.outcome() is None and (move_count <n_moves):
            r =  np.random.uniform(0,1)
            if r > 0.25:
                boards.append(copy.deepcopy(board))
                pos_count += 1
                pbar.update(1)
            p = 0.05 if move_count<20 or move_count>100 else 0.9
            if r > p:
                _, move = get_stockfish_value_moves([board], time_limit=0.001, return_policies=False)
                move = np.random.choice(move)
            else:
                move = np.random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1
    pbar.close()
    if policy_format == 'both':
        values, graph_policies, array_policies = get_stockfish_value_moves(boards, policy_format=policy_format, return_policies=True, time_limit=0.01, disable=False)
        return boards, values, graph_policies, array_policies
    else:
        values, policies = get_stockfish_value_moves(boards, policy_format=policy_format, return_policies=True, time_limit=0.01, disable=False)
        return boards, values, policies

def custom_collate_fn(inpt):
    print(inpt)
    raise
def load_data(path,policy_format='graph',stockfish_value=True,testing=False,testing_range=(0,1000),save_path=None,time_limit=0.1,n_random_pos=0):
    games_df = pd.read_csv(path)
    games_df = games_df[~games_df['result'].isna()]
    buffer = Buffer(max_size=np.inf, policy_format=policy_format)
    boards, values, policies = [], [], []
    graph_policies, array_policies = [], []
    df =   games_df.iloc[testing_range[0]:testing_range[1]] if testing else games_df
    for idx, row in tqdm(df.iterrows()):
        pgn_string = row['moves']
        result = row['result']
        if policy_format == 'both':
            board, value, graph_policy, array_policy = encode_game(pgn_string,result,stockfish_value,time_limit=time_limit,policy_format=policy_format)
            boards.extend(board)
            values.extend(value)
            graph_policies.extend(graph_policy)
            array_policies.extend(array_policy)
        else:
            board, value, policy  = encode_game(pgn_string,result,stockfish_value,time_limit=time_limit,policy_format=policy_format)
            boards.extend(board)
            values.extend(value)
            policies.extend(policy)
    if policy_format == 'both':
        buffer.push(boards, values, None,graph_policies, array_policies)
    else:
        buffer.push(boards, values, policies)
    if n_random_pos > 0:
        if policy_format == 'both':
            del boards, values, graph_policies, array_policies
            boards, values, graph_policies, array_policies = gen_positions(n_random_pos,policy_format=policy_format)
            buffer.push(boards, values, None,graph_policies, array_policies)
        else:
            del boards, values, policies
            boards, values, policies = gen_positions(n_random_pos,policy_format=policy_format)
            buffer.push(boards, values, policies)
    if save_path is not None:
        buffer.save(save_path)
    return buffer


if __name__ == '__main__':
    # Testing
    testing_range = (0,1)
    data_path = '/Users/mrsalwer/Desktop/graph_chess/train_human_games/human_games/games_chesstempo.csv'
    buf = load_data(path=data_path,
                    testing=True,
                    stockfish_value=True,
                    testing_range=testing_range,
                    save_path='buffers/{}_{}_games_stock_70krandom'.format(testing_range[0],testing_range[1]),
                    time_limit=0.01,
                    n_random_pos=2)


    # for testing_range in ranges:
        # buf = load_data()
        # train network on data for 1 epoch
        # save network
