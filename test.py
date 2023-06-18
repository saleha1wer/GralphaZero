import chess.engine
import numpy as np
from utils.action_encoding import encode_action
import chess

from torch.nn.functional import cross_entropy
import torch

# a = torch.tensor([0,0,0],dtype=torch.float)
# b = torch.tensor([1,1,1],dtype=torch.float)

# c = torch.tensor([0.5,0.5,0.5],dtype=torch.float)
# d = torch.tensor([0.5,0.5,0.5],dtype=torch.float)


# print(cross_entropy(a,b)+cross_entropy(c,d))


# a = torch.tensor([0,0,0,0.5,0.5,0.5],dtype=torch.float)
# b = torch.tensor([1,1,1,0.5,0.5,0.5],dtype=torch.float)
# print(cross_entropy(a,b))

# async def analyze_boards(boards):
#     transport, engine = await chess.engine.popen_uci(r'/opt/homebrew/opt/stockfish/bin/stockfish')
#     scores = []
#     best_moves = []
#     for board in boards:
#         info = await engine.analyse(board, chess.engine.Limit(time=0.1))
#         score = info['score'].white().score(mate_score=10000)
#         moves = info['pv']
#         move = info['pv'][0] if len(moves) < 2 else moves[:2]
#         best_moves.append(move)
#         if score in [10000,-10000]:
#             score = 1 if score > 0 else -1
#         else:
#             adjusted_score = score/1000
#             score = min(adjusted_score,1) if score > 0 else max(adjusted_score,-1)
#         scores.append(score)
#     await engine.quit()
#     return scores,best_moves
# def get_stockfish_values_policies(boards):
#     import asyncio
#     asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
#     scores,best_moves = asyncio.run(analyze_boards(boards))
#     new_policies = []
#     for idx,board in enumerate(boards):                       
#         npolicy = np.zeros((8,8,73))
#         move = best_moves[idx]
#         if move.__class__ == list:
#             act1_idx = encode_action(board,move[0])
#             act2_idx = encode_action(board,move[1])
#             act1_pol = np.random.uniform(0.5,0.8)
#             npolicy[act1_idx[0],act1_idx[1],act1_idx[2]] =act1_pol
#             npolicy[act2_idx[0],act2_idx[1],act2_idx[2]] = 1-act1_pol
#         else:
#             act_idx = encode_action(board,move)
#             npolicy[act_idx[0],act_idx[1],act_idx[2]] = 1
#         new_policies.append(npolicy)
#     return scores,new_policies

# fens = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnq/pppppppp/8/8/8/8/PPPPPPPP/RN2KBNR b KQq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1']
# boards = []
# for i in range(5):
#     board = chess.Board(fens[i])
#     boards.append(board)

# values,policies = get_stockfish_values_policies(boards)
# print(len(values))
# print(len(policies))
