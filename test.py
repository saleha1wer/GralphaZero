# import pickle 

# with open('root.pkl', 'rb') as inp:
#     root = pickle.load(inp)

# print(root.ave_eval())
# for i in root.children:
#     print(i.move, ': ')
#     print('n_visits:', i.n_visits)
#     # print('evals:', i.evals)
#     # print('Prior:', i.P)
#     # print('UCB:', i.ucb(2))
#     # for z in i.children:
#     #     print(z.move, ': ')
#     #     print('Prior:', i.P)

import chess
# import copy


board = chess.Board()
# board.push(chess.Move.from_uci('e2e4'))
# board.push(chess.Move.from_uci('e7e5'))
# board_copy = chess.Board(board.fen())
# print(board_copy.fullmove_number)

import torch
from utils.board2graph import board2graph
import numpy as np
from utils.action_encoding import decode_action,encode_action
model = torch.load('final_net')
model.eval()
value, policy = model([board2graph(board)])
value = value[0].detach().numpy()
policy = policy[0].detach().numpy()
print(value)
idx = np.unravel_index(np.argmax(policy), (8,8,73))
print(idx)
print(decode_action(board,policy))


