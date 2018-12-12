# deep-briscola

Tensorflow deep reinforcement learning agent playing Briscola card game.

[**What is Briscola??**](https://en.wikipedia.org/wiki/Briscola)


This repository contains a Briscola game environment where different agents can play.

 - `RandomAgent`: choose each move in a random fashion
 - `AIAgent`: knows the rules and the strategies for winning the game
 - `DeepAgent`: agent trained using deep reinforcement learning
 - `HumanAgent`: yourself


## Train a model

    $ python train.py --saved_model saved_model_dir

## Play against AI Agent

    $ python human_vs_ai.py

## Play against trained deep agent

    $ python human_vs_ai.py --saved_model saved_model_dir