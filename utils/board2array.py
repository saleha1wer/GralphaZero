import chess
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
def board2array(board: chess.Board):
    encoded = np.zeros([8,8,21]).astype(int)
    encoded_dict = {"R":0, "N":1, "B":2, "Q":3, "K":4, "P":5, "r":6, "n":7, "b":8, "q":9, "k":10, "p":11}
    for i in range(8):
        for j in range(8):
            idx = (i,j) if board.turn else (7-i,7-j)
            piece_str = str(board.piece_at(chess.square(idx[0],idx[1])))
            if piece_str != 'None':
                encoded[i,j,encoded_dict[piece_str]] = 1

    if board.turn is chess.WHITE:
        encoded[:,:,12] = 1 # player to move

    if not board.has_kingside_castling_rights(chess.WHITE):
        encoded[:,:,13] = 1
    if not board.has_queenside_castling_rights(chess.WHITE):
        encoded[:,:,14] = 1
    if not board.has_kingside_castling_rights(chess.BLACK):
        encoded[:,:,15] = 1
    if not board.has_queenside_castling_rights(chess.BLACK):
        encoded[:,:,16] = 1

    encoded[:,:,17] = board.fullmove_number
    if (not board.is_repetition(2)) and (not board.is_repetition(3)):
        v = 0 
    else:
        v = 1 if board.is_repetition(2) else 2
    encoded[:,:,18] = v
    encoded[:,:,19] = board.halfmove_clock
    encoded[:,:,20] = -1 if board.ep_square is None else board.ep_square % 8
    return encoded