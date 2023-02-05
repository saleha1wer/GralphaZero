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
  - [ ] Add to rep: 1. side to move, 2. repetition count, 3. move count and 4. no progress count 
- [x] Network
- [ ] MCTS 
- [ ] Training 
  - [x] buffer
  - [ ] encode/decode action from/to (8,8,73) matrix
  - [ ] Gather data with self-play (MCTS)
  - [ ] Train network on random batch of data
  - [ ] Evaluate against previous network (400 games), pick best network (ran every 1000 training loops)
  - [ ] Function to calculate (and store) Elo of network (ran every 100 training loops)
- [ ] Play  
  - [ ] Load policy network
  - [ ] Run 1600 simulations of MCTS and select the child node with highest N value.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Saleh Alwer - saleh.tarek.issa.alwer@umail.leidenuniv.nl


