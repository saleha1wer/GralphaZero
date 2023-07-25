"""
    - Load pretrained model
    - Load player specific data
    - Calc loss of pretrained model on player specific data
    - Finetune model on data
    - Calc loss of finetuned model on player specific data
    - Save model
"""
from network import Network
import torch
import numpy as np
import chess.pgn
import chess
import copy
from utils.action_encoding import get_num_edges, encode_action
from utils.board2graph import board2graph
from tqdm import tqdm
from datamodule import ChessDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

def load_player_data(data_paths,white_data_first):
    assert white_data_first
    boards,policies = [],[] 
    for idx,data_path in enumerate(data_paths): # data paths contains paths to white and black data
        pgn_file = open(data_path)
        games = []
        i = 0
        while True and i<1: # Load all games from pgn file
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
            i+=1
        pgn_file.close()
         
        for game in tqdm(games,desc='Extracting boards and policies from games'): # Extract boards and policies from games
            board = chess.Board()
            for move in game.mainline_moves():
                if (board.turn == chess.WHITE and idx == 0) or (board.turn == chess.BLACK and idx == 1):
                    temp_board = copy.deepcopy(board)
                    boards.append(temp_board)
                    policy = np.zeros(get_num_edges(temp_board))
                    policy[encode_action(temp_board,move)] = 1
                    policies.append(policy)
                board.push(move)
        # make chess dataset
    return boards, policies 

def eval(model, data):
    """
    Evaluates model (only policy) on data given (data should have boards and policies)
    Returns Accuracy
    """
    model.eval()
    boards,policies = data
    # boards = [board2graph(board) for board in tqdm(boards,desc='Converting boards to graphs')]
    # acc = 0
    # for board,policy in tqdm(zip(boards,policies),desc='Evaluating model'):
        # _,pred = model([board])
        # pred = pred[0].detach().view(-1).numpy()
        # if np.argmax(pred) == np.argmax(policy):
            # acc+=1
    # acc = acc/len(boards)
    # print(f'Accuracy: {acc}')
    data = ChessDataset(boards=boards,values=np.full(len(policies),np.nan),policies=policies)
    dataloader = DataLoader(data,batch_size=min(256,int(len(boards)/4)))
    trainer = pl.Trainer(accelerator='cpu', devices=1)
    res = trainer.test(model,dataloader)
    del model
    return res

def _finetune(model,train_data,val_data,epochs,device,name,batch_size=32):
    model.train()
    train_boards,train_policies = train_data
    val_boards,val_policies = val_data
    model.freeze_model()
    model.unfreeze_model_parts()
    model.freeze_gnn_edge(fraction=0.5)
    # make data
    train_data = ChessDataset(boards=train_boards,values=np.full(len(train_policies),np.nan),policies=train_policies)
    val_data = ChessDataset(boards=val_boards,values=np.full(len(val_policies),np.nan),policies=val_policies)
    # make dataloader 
    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    val_dataloader = DataLoader(val_data,batch_size=batch_size)
    # train
    logger = TensorBoardLogger("fintune_logs",name=name)
    trainer = Trainer(accelerator=device, devices=1, max_epochs=epochs,logger=logger,log_every_n_steps=10)
    trainer.fit(model,train_dataloader,val_dataloaders=val_dataloader)
    return model

def finetune(model_config,model_params, player_data_paths,epochs=5,name='temp',batch_size=32):
    model_config['finetune'] = True
    model = Network(model_config)
    model.load_state_dict(torch.load(model_params))
    boards, policies = load_player_data(player_data_paths,white_data_first=True)
    val_boards, val_policies = boards[:int(len(boards)*0.2)], policies[:int(len(policies)*0.2)]
    train_boards, train_policies = boards[int(len(boards)*0.2):], policies[int(len(policies)*0.2):]
    pretrain_acc = eval(model, (val_boards,val_policies))
    print(f'Pretrain Results (Val):\n {pretrain_acc}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _finetune(model,(train_boards,train_policies),(val_boards,val_policies),epochs,device,name,batch_size=batch_size)
    save_path = 'finetuned_networks/finetuned_network_'+name
    torch.save(model.state_dict(), save_path)
    finetune_acc =eval(model, (val_boards,val_policies))
    print(f'Finetuned Results (Val):\n {finetune_acc}')


if __name__ == '__main__':
    model_config = {'board_representation':'graph','policy_format':'graph','lr': 0.00003,'GAheads': 16, 'att_emb_size': 1024, 'heads_GAT_edge': 16, 'heads_GAT_graph': 256, 'hidden_edge': 2048, 'hidden_graph': 512, 'n_layers': 5, 'pol_nlayers': 5, 'value_nlayers': 5}
    model_params = 'networks/graph_graph_15k1200k_params'
    names = ['saleh', 'firouzja','karpov']
    for name in names:
        player_data_paths = [f'train_human_games/human_games/{name}-white.pgn', f'train_human_games/human_games/{name}-black.pgn']
        finetune(model_config,model_params,player_data_paths,name=name,epochs=5,batch_size=64)
