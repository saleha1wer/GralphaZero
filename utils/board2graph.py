import chess
import numpy as np
import torch
import copy
from torch_geometric.data import Data

def encode_piece_node(board,square): 
    """ 
    Returns one-hot encoding of a chess piece
    """
    piece = board.piece_at(square)
    color = 0
    space = np.zeros(25)
    space[13] = 1 if board.turn else -1
    if board.turn == chess.WHITE:
        space[14] = int(square//8)
        space[15] = int(square%8)
    else:
        space[14] = int((63-square)//8)
        space[15] = int((63-square)%8)
    space[16] = 1 if board.is_repetition(2) else 0
    space[17] = 1 if board.is_repetition(3) else 0
    space[18] = board.fullmove_number
    space[19] = board.halfmove_clock
    space[20] = -1 if board.ep_square is None else board.ep_square % 8
    if not board.has_kingside_castling_rights(chess.WHITE):
        space[21] = 1
    if not board.has_queenside_castling_rights(chess.WHITE):
        space[22] = 1
    if not board.has_kingside_castling_rights(chess.BLACK):
        space[23] = 1
    if not board.has_queenside_castling_rights(chess.BLACK):
        space[24] = 1
    if piece == None:
        space[0] = 1
        return torch.tensor(space,dtype=torch.float).view(-1, 25)
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
    return torch.tensor(space,dtype=torch.long).view(-1, 25)


def encode_move_edge(move): 
    """ 
    Returns the edge of move. The edge connection is defined by the squares
    from and to which a move is legal.
    """
    return [move.from_square, move.to_square]


def board2graph(board: chess.Board, policy=None, pred_policy=None,return_nodes=False):
    """ 
    Encodes a chess board into a graph with the needed structure. Each square is a node. An edge implies a legal move from one square to the other.

    Node features : 0 if square is empty, 1 for pawn, 2 for bishop, 3 for knight, 4 for rook, 5 for queen, 6 for king. One hot encoded.

    Edge List: [from_square, to_square] for each legal move for both players.

    Edge features: [1,0] for the current players moves, [0,0] for the opponent's legal moves and [1,n] or [0,n] for the previous n moves.
    """
    # encode pieces
    nodes = [i for i in range(64)]
    node_features = [encode_piece_node(board, node) for node in nodes]
    
    edge_list = []
    edge_features = []
    
    # encode side to move moves
    for move in board.legal_moves:
        edge_list.append(encode_move_edge(move))
        edge_features.append([1,0,0])
    
    # encode side not to move moves
    opp_turn = copy.deepcopy(board)
    opp_turn.turn = not opp_turn.turn
    for move in opp_turn.legal_moves:
        edge_list.append(encode_move_edge(move))
        edge_features.append([0,0,0])

    # encode side to move moves
    moves_list = [encode_move_edge(move) for move in board.legal_moves]

    # encode edge features
    edge_features = [[1,0,0] if edge in moves_list else [0,0,0] for edge in edge_list]

    # dictionary to keep track of move count
    count_dict = {str(move)[:4]: 0 for move in board.legal_moves}

    # handle promotions
    for move in board.legal_moves:
        if move.promotion is not None:
            idxs = np.where(np.array(edge_list) == encode_move_edge(move))[0]
            idx = idxs[count_dict[str(move)[:4]]]
            edge_features[idx][2] = int(move.promotion)
            count_dict[str(move)[:4]] += 1

    count_dict = {str(move)[:4]: 0 for move in opp_turn.legal_moves}
    for move in opp_turn.legal_moves:
        if move.promotion is not None:
            idxs = np.where(np.array(edge_list) == encode_move_edge(move))[0]
            idx = idxs[count_dict[str(move)[:4]]]
            edge_features[idx][2] = int(move.promotion)
            count_dict[str(move)[:4]] += 1

    # encode n previous moves 
    prev_moves_board = copy.deepcopy(board)
    for i in range(1,6):
        try:
            last_move = prev_moves_board.pop()
        except IndexError: # no more moves
            break
        edge_list.append(encode_move_edge(last_move))
        turn = 1 if prev_moves_board.turn == board.turn else 0
        p = 0 if last_move.promotion is None else int(last_move.promotion)
        edge_features.append([turn, i, p])

    # handle policy if provided (during training, this is where we keep the target policy)
    if policy is not None:
        for idx, edge_at in enumerate(edge_features):
            p = -1 if (edge_at[0] != 1) or (edge_at[1]>0) else policy[idx]
            edge_at.append(p)
    if pred_policy is not None:
        for idx, edge_at in enumerate(edge_features):
            p = -1 if (edge_at[0] != 1) or (edge_at[1]>0) else pred_policy[idx]
            edge_at.append(p)
    if return_nodes:
        return nodes, Data(x=torch.stack(node_features,dim=0).view(len(nodes), 25), edge_index=torch.tensor(edge_list, dtype=torch.int64).t().view(2, -1), edge_attr=torch.tensor(edge_features, dtype=torch.float))
    return Data(x=torch.stack(node_features,dim=0).view(len(nodes), 25), edge_index=torch.tensor(edge_list, dtype=torch.int64).t().view(2, -1), edge_attr=torch.tensor(edge_features, dtype=torch.float))
        # return node_features, edge_list,edge_features


# b = chess.Board()
# b.push_san("e4")
# b.push_san("e5")
# b.push_san("Nf3")
# b.push_san("Nc6")
# b.push_san("Bb5")
# g = board2graph(b,policy=np.random.uniform(size=30))
# print(g.x.shape)
# print(g.edge_index.shape)
# print(g.edge_attr)