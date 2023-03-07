# Graph-AlphaZero 
<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to train an alphazero-style engine for Chess. The biggest difference being the policy network; we use a [Graph Attention Network](https://arxiv.org/abs/1710.10903)
### Built With

* Python
* PyTorch Geometric
* Python-Chess

<!-- ROADMAP -->
## Roadmap
- [x] Graph representation
  - [x] Chess board --> Graph
  - [x] Add to rep: 1. side to move, 2. repetition count, 3. move count and 4. no progress count 
- [x] Network
- [x] MCTS 
- [x] Training 
  - [x] Buffer
  - [x] Encode/decode action to/from (8,8,73) matrix
  - [x] Gather data with self-play (MCTS)
  - [x] Train network on random batch of data
  - [ ] Evaluate against previous network (400 games), pick best network 
  - [x] Function to calculate (and store) Elo of network
- [x] Play  
  - [x] Load policy network
  - [x] Run 1600 simulations of MCTS and select the child node with highest N value.
- [x] Multiprocessing 

<!-- CONTACT -->
## Contact

Saleh Alwer - saleh.tarek.issa.alwer@umail.leidenuniv.nl


