# deep-briscola

Tensorflow deep reinforcement learning agent playing Briscola card game.

[**What is Briscola??**](https://en.wikipedia.org/wiki/Briscola)


This repository contains a Briscola game environment where different agents can play.

 - `RandomAgent`: choose each move in a random fashion
 - `AIAgent`: knows the rules and the strategies for winning the game
 - `DeepAgent`: agent trained using deep reinforcement learning
 - `HumanAgent`: yourself


## Dependencies

```
sudo apt-get update && sudo apt-get install -y \
  python-dev \
  python3-pip
  
sudo pip install \
  tensorflow \
  hyperopt \
  matplotlib \
  pandas

```

Alternatively, a Dockerfile with all the dependencies installed is provided in this repo.
To use it:

[Install Docker on Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

```
$ bash docker/build.sh
$ bash docker/run.sh
```

## Usage

##### Train a model

    $ python3 train.py --saved_model saved_model_dir

##### Play against trained deep agent

    $ python3 human_vs_ai.py --saved_model saved_model_dir

##### Play against AI Agent

    $ python3 human_vs_ai.py


## Features

##### Different networks implemented

Specify the network type using `--network` command line argument

 - Deep Q Network
 - Deep Recurrent Q Network
 - WIP Synchronous Advantage Actor Critic (A2C)

##### Self Play

Train multiple agents using the `self_train.py` python script.

## Results

 - Training a Deep Q Network model for 75k epochs: achieved 85% winrate against a random player.

 - Training a Deep Q Network model for 100k epochs using Self Play: achieved 90% winrate against a random player.
