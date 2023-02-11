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
import copy

board = chess.Board()
board.push(chess.Move.from_uci('e2e4'))
board.push(chess.Move.from_uci('e7e5'))
board_copy = chess.Board(board.fen())
print(board_copy.fullmove_number)
