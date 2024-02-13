# flappybird-dqn-agent

A DQN (Deep Q-Network) agent has been developed to play Flappy Bird across two varying levels of difficulty. The primary objective for this AI agent was to achieve an average score of over 500 in a less challenging environment and over 100 in a more challenging one.


## Description of environments


### [Flappybird](https://github.com/yenchenlin/DeepLearningFlappyBird) 

This version is a Flappy Bird clone created using PyGame, adapted from a repository found on GitHub. This environment is considered less challenging compared to its counterpart, primarily due to the smaller height differences between successive pipes and a more uniform distribution of obstacles.


### [Flappy bird gymnasium](https://pypi.org/project/flappy-bird-gymnasium/)

This environment presents a higher difficulty level for the Flappy Bird game. It is accessible through the flappy-bird-gymnasium package on PyPI. The challenge here stems from larger height variances between consecutive pipes and a more erratic distribution throughout the game, making it tougher for the AI agent.


## DQN Agent

The [DQN architecture](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) was inspired by the Deep Mind Agent's trained to play Atari.

## Performances

I trained two models:

- `trained_0_1600000.pth` - A model trained on 1.6 million frames in the flappy-bird-gymnasium environment, achieving an average score of 121 over 30 games.
- `trained_1_1450000.pth` - A model trained on 1.45 million frames in the Flappybird environment, achieving an average score of 748 over 30 games.

I have used the dedicated GPU on a MacbookPro 2020, M1 CPU with 8 GB of GPU. 
The training have lasted somewhere around 9 hours. On a faster GPU, might take less. On CPU, might take some days.


## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015 
