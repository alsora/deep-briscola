import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil

from networks.dqn import DQN
from networks.drqn import DRQN


class QAgent():
    ''' Trainable agent which uses a neural network to determine best action'''

    def __init__(self, epsilon=0.85, epsilon_increment=0, epsilon_max = 0.85, discount=0.95, learning_rate = 1e-3):
        self.n_actions = 9
        self.n_features = 27
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon
        self.epsilon_backup = epsilon
        self.epsilon_increment = epsilon_increment
        self.gamma = discount

        self.last_state = None
        self.state = None
        self.action = None
        self.reward = None

        # create q learning algorithm
        self.q_learning = DQN(self.n_actions, self.n_features, learning_rate, discount)

        self.last_wrong_move = False
        self.count_wrong_moves = 0


    def observe(self, game, player_id):
        self.board = game.board
        self.id = player_id

        state=np.zeros(self.n_features)
        # add my board
        state[:9] = [1 if x == player_id else 0 for x in self.board]
        # add opponent board
        state[9:18] = [1 if x == (1 -player_id) else 0 for x in self.board]
        # add complete board
        state[18:27] = self.board

        self.last_state = self.state
        self.state = state



    def select_action(self, available_actions):
        '''Selects an action given the observed state'''

        if self.state is None:
            raise ValueError("DeepAgent.select_action called before observing the state")
        #print (self.state)
        #print(self.epsilon)
        if np.random.uniform() > self.epsilon:
            # select action randomly with probability (1 - epsilon)
            action = np.random.choice(available_actions)
        else:
            q = self.q_learning.get_q_table(self.state)
            # sort actions from highest to lowest predicted q value
            sorted_actions = (-q).argsort()

            for predicted_action in sorted_actions:
                if predicted_action in available_actions:
                    action = predicted_action
                    break

            if action != sorted_actions[0]:
                self.last_wrong_move = True
                self.count_wrong_moves += 1
                action = np.random.choice(available_actions)

            if self.count_wrong_moves == 500:
                print ("500 wrong moves!")
                self.count_wrong_moves = 0

        # store the chosen action
        self.action = action
        return action


    def update(self, reward):
        ''' After receiving a reward the agent has all collected [s, a, r, s_]'''

        if self.last_wrong_move:
            # reduce the reward if the agent's last move has been a not allowed move
            self.reward = -10
            self.last_wrong_move = False
        else:
            self.reward = reward

        # update last reward
        self.reward = reward
        # update epsilon grediness
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.q_learning.learn(self.last_state, self.action, self.reward, self.state)


    def save_model(self, output_dir):
        self.q_learning.save_model(output_dir)

    def load_model(self, saved_model_dir):
        self.q_learning.load_model(saved_model_dir)

    def make_greedy(self):
        self.epsilon_backup = self.epsilon
        self.epsilon = 1.0

    def restore_epsilon(self):
        self.epsilon = self.epsilon_backup