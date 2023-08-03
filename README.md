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
  - [x] Board --> Graph
  - [x] Policy graph representation
- [x] Network
- [x] Train on human/random games with Stockfish evaluations
- [ ] Train on MCTS data (selfplay)
