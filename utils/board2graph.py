from platform import node
import chess
from pyrsistent import b
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import typing
import matplotlib
import copy
from torch_geometric.data import Data


class Graphs:
    def __init__(self, node_features: torch.Tensor,
                 edge_list: torch.Tensor, edge_features: torch.Tensor):
        """
        A graph datastructure which groups together the series of tensors that represent the
        graph. Note that this datastructure also holds multiple molecule graphs as one large
        disconnected graph -- the nodes belonging to each molecule are described by node_to_graph_id.

        ## Further details on the individual tensors
        Say this graph represents acetone, CC(=O)C, and ethane, CC, and we're using a simple
        three dimensional one-hot encoding for 'C', 'O' and 'N' and a simple two dimensional
        one-hot encoding for the bonds 'SINGLE', 'DOUBLE' then the resulting tensors would look
        like:
        node_features = [[1. 0. 0.],
                         [1. 0. 0.],
                         [0. 1. 0.],
                         [1. 0. 0.],
                         [1. 0. 0.],
                         [1. 0. 0.]]
        edge_list = [[0 1],
                     [1 0],
                     [1 2],
                     [2 1],
                     [1 3],
                     [3 1],
                     [4 5],
                     [5 4]]

        edge_features = [[1. 0.],
                         [1. 0.],
                         [0. 1.],
                         [0. 1.],
                         [1. 0.],
                         [1. 0.],
                         [1. 0.],
                         [1. 0.]]

        More generally we expect the different tensors to have the following datatypes and shapes
        (below N is number of nodes, E number of edges, h_n the feature dimensionality of node
        features and h_e the feature dimensionality of edge features):

        :param node_features: Tensor (dtype float32 , shape [N, h_n])
        :param edge_list: Tensor (dtype int64 , shape [E, 2])
        :param edge_features: Tensor (dtype float32 , shape [E, h_e])
        """
        self.node_features = node_features
        self.edge_list = edge_list
        self.edge_features = edge_features

    def to(self, *args, **kwargs):
        """
        Works in a similar way to the Tensor function torch.Tensor.to(...)
        and performs  dtype and/or device conversion for the entire datastructure
        """
        new_graph = type(self)(self.node_features.to(*args, **kwargs),
                               self.edge_list.to(*args, **kwargs),
                               self.edge_features.to(*args, **kwargs)
                               )
        return new_graph

    @classmethod
    def from_board(cls, board):
        """
        Converts a board object into the representation required by this datastructure.
        """
        # Convert to form we need using previous code:
        node_features, edge_list, edge_features = board2graph(board)


        # Convert to tensors:
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_list = torch.tensor(edge_list, dtype=torch.int64)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        return cls(node_features, edge_list, edge_features)


def encode_piece_node(piece): 
    """ 
    Returns one-hot encoding of a chess piece
    """
    color = 0
    space = np.zeros(13)
    if piece == None:
        space[0] = 1
        return torch.tensor(space,dtype=torch.float).view(-1, 13)
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
    return torch.tensor(space,dtype=torch.long).view(-1, 13)


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

    Edge List: List of legal moves in the format of connecting the nodes representing the 'sqaure from' node to the 'square to' node
    """
    node_features = [encode_piece_node(board.piece_at(i)) for i in range(64)]
    edge_list = [encode_move_edge(move) for move in board.legal_moves]

    opp_turn = copy.deepcopy(board)

    opp_turn.turn = not opp_turn.turn
    edge_list.extend([encode_move_edge(move) for move in opp_turn.legal_moves])
    
    moves_list = [encode_move_edge(move) for move in board.legal_moves]
    edge_features = [[0] for i in range(len(edge_list))]
    for move in moves_list:
        edge_features[edge_list.index(move)] = [1]
    return Data(x=torch.stack(node_features,dim=0).reshape(64, 13), edge_index=torch.tensor(edge_list, dtype=torch.int64).t().view(2, -1), edge_attr=torch.tensor(edge_features, dtype=torch.float))
    # return node_features, edge_list,edge_features




