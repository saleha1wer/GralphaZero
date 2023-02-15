import chess
import numpy as np
import torch
import copy
from torch_geometric.data import Data

def encode_piece_node(piece,board,square): 
    """ 
    Returns one-hot encoding of a chess piece
    """
    color = 0
    space = np.zeros(20)
    space[13] = 1 if board.turn else -1
    space[14] = int(square//8)
    space[15] = int(square%8)
    space[16] = 1 if board.is_repetition(2) else 0
    space[17] = 1 if board.is_repetition(3) else 0
    space[18] = board.fullmove_number
    space[19] = board.halfmove_clock
    if piece == None:
        space[0] = 1
        return torch.tensor(space,dtype=torch.float).view(-1, 20)
    if piece.color != chess.WHITE:
        color = 6
    if piece.piece_type == chess.PAWN:
        idx = 1
    elif piece.piece_type == chess.BISHOP:
        idx = 2
    elif piece.piece_type == chess.KNIGHT:
        idx = 3
    elif piece.piece_type == chess.ROOK:
        idx = 4
    elif piece.piece_type == chess.QUEEN:
        idx = 5
    elif piece.piece_type == chess.KING:
        idx = 6
    space[idx+color] = 1
    return torch.tensor(space,dtype=torch.long).view(-1, 20)


def encode_move_edge(move): 
    """ 
    Returns the edge of move in string format 'a2a4' becomes the edge connection
    """
    to_sq = move.to_square
    from_sq = move.from_square
    return [from_sq, to_sq]


def board2graph(board: chess.Board):
    """ 
    Encodes a chess board into a graph with the needed structure. Each square is a node. An edge implies a legal move from one square to the other.

    Node features : 0 if square is empty, 1 for pawn, 2 for bishop, 3 for knight, 4 for rook, 5 for queen, 6 for king. One hot encoded.

    Edge List: [from_square, to_square] for each legal move for the both players

    Edge features: [1,0] for the current players moves, [0,0] for the opponent's legal moves and [1,n] or [0,n] for the previous n movees.
    """
    node_features = [encode_piece_node(board.piece_at(i),board,i) for i in range(64)]
    edge_list = [encode_move_edge(move) for move in board.legal_moves]

    opp_turn = copy.deepcopy(board)

    opp_turn.turn = not opp_turn.turn
    edge_list.extend([encode_move_edge(move) for move in opp_turn.legal_moves])
    
    moves_list = [encode_move_edge(move) for move in board.legal_moves]
    edge_features = [[0] for i in range(len(edge_list))] 
    for move in moves_list:
        edge_features[edge_list.index(move)] = [1]
    
    return Data(x=torch.stack(node_features,dim=0).reshape(64, 20), edge_index=torch.tensor(edge_list, dtype=torch.int64).t().view(2, -1), edge_attr=torch.tensor(edge_features, dtype=torch.float))
    # return node_features, edge_list,edge_features




