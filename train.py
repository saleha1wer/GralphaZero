from mcts import MCTS_selfplay
import pytorch_lightning as pl
from eval import find_network_elo
import torch
from tqdm import tqdm
from buffer import Buffer, join_buffers
from torch_geometric.loader import DataLoader
from network import GNN
import multiprocessing as mp
import time 
"""
Repeat (500 times) - 2,500,000 games of self-play data
    - Gather 5,000 games of self-play data
        Store policies, values for each board in buffer of size 10,000,000 
    - Train network (1024 batch size, 5 epochs) 
    - Evaluate against previous network (Every 75 loops) -todo
        Replace best network if wins 55% of games (and save network)
    - Calculate elo of network (Every 50 loops)
"""
# network = GNN({'lr': 0.05, 'hidden': 4672, 'n_layers': 2})
# network.eval()
# def net_forward(graph_obj):
#     return network(graph_obj)

def decrease_c(max_c, min_c,loop_num,threshold,decay):
    if loop_num < threshold:
        return max_c

    return max(min_c, max_c*((1-decay)**(loop_num-threshold)))


def train(network_params_path, n_loops=500,n_games_per_loop=5000, n_sims_per_move=1600,sample_size = 500000,buffer_size=7500000,batch_size=1024, eval_freq=5, calc_elo_freq=50,disable_game_bar=False,disable_mcts_bar=True,stockfish=False,elo_threshold=55,elo_decay=0.5, c_max=1.5, c_min=0.5, c_decay=0.5, c_threshold=75):
    main_buffer = Buffer(max_size=buffer_size)
    elo_ratings = []
    config = {'lr': 0.001 , 'hidden': 4672, 'n_layers': 1,'heads': 4}
    network = GNN(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if network_params_path is not None:
        start = time.time()
        print('Loading Network from: ', network_params_path)
        network.load_state_dict(torch.load(network_params_path))
        print('Loaded network from: ', network_params_path)
        print('Time taken: ', time.time() - start)
    # n_cpu = mp.cpu_count()
    # n_games_per_loop = int(n_games_per_loop/n_cpu)
    n_cpu = 5
    n_games_per_loop = int(n_games_per_loop/n_cpu)
    for i in tqdm(range(1,n_loops+1)):
        if i > 1:
            network = GNN(config)
            network.load_state_dict(torch.load('temp_network_selfplay_params'))
        network.eval()
        c = decrease_c(max_c=2,min_c=0.4,loop_num=i,threshold=100,decay=0.002)
        print('Number of CPUs: ', n_cpu)
        pool = mp.Pool(processes=n_cpu)
        network.to(device)
        # Define a list of input tuples for each process
        inputs = [(network, c, n_games_per_loop, n_sims_per_move, None,disable_game_bar,disable_mcts_bar,stockfish) for _ in range(pool._processes)]
        # Run the function in parallel using the pool of processes
        buffers = pool.starmap(MCTS_selfplay, inputs)
        pool.close()
        pool.join()
        network.train()
        # Join the buffers
        buffer_df = join_buffers(buffers)
        main_buffer.push_df(buffer_df)
        print('Buffer size: ', main_buffer.__len__())
        # train network on random batch of data
        data = main_buffer.sample(sample_size) 
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=2)
        trainer.fit(network, dataloader)
        torch.save(network.state_dict(), 'temp_network_selfplay_params')
        # Calculate elo of network
        if i % calc_elo_freq == 0:
            network.eval()
            elo = find_network_elo(network,num_games=100,num_runs=n_sims_per_move,save_pgn_path='PGN_eval_{}.txt'.format(i))
            torch.save(network, 'net_{}'.format(i))
            elo_ratings.append(elo)
            print('Elo after {} loops: {}'.format(i,elo))
    print('Elo ratings: ', elo_ratings)

    # Save network parameters
    torch.save(network.state_dict(), 'network_selfplay_params')


if __name__ == '__main__':
    # Train for 1,000,000 games
    # train(n_loops=2000, 
    #     n_games_per_loop=500, 
    #     n_sims_per_move=1600, 
    #     buffer_size=250000,
    #     sample_size=25000, 
    #     batch_size=1024, 
    #     eval_freq=75, 
    #     calc_elo_freq=100,
    #     disable_game_bar=False,
    #     disable_mcts_bar =True)

    #Test
    network_params_path = None
    network_params_path = 'network_human_games_params'
    train(network_params_path=network_params_path,
          n_loops=5,
          n_games_per_loop=5, 
          n_sims_per_move=2,
          buffer_size=1600,
          sample_size=800, 
          batch_size=32, 
          eval_freq=75, 
          calc_elo_freq=10,
          disable_game_bar=False,
          disable_mcts_bar=True,
          stockfish = True)