import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil

from networks.dqn import DQN
from networks.drqn import DRQN
from utils import NetworkTypes

class QAgent():
    ''' Trainable agent which uses a neural network to determine best action'''

    def __init__(self, epsilon=0.85, epsilon_increment=0, epsilon_max=0.85, discount=0.95, network=NetworkTypes.DRQN, layers=[256, 128], learning_rate=1e-3, replace_target_iter=2000, batch_size=100):
        self.name = 'QAgent'

        self.n_actions = 3
        self.n_features = 70
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon
        self.epsilon_backup = epsilon
        self.epsilon_increment = epsilon_increment

        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None
        self.terminal = None
        self.network = network

        # create q learning algorithm
        if network == NetworkTypes.DQN:
            self.q_learning = DQN(self.n_actions, self.n_features, layers, learning_rate, batch_size, replace_target_iter, discount)
        elif network == NetworkTypes.DRQN:
            self.q_learning = DRQN(self.n_actions, self.n_features, layers, learning_rate, batch_size, replace_target_iter, discount)
        else:
            raise ValueError("Not implemented type of network passed to QAgent")


    def observe(self, game, player):
        ''' create an encoded state representation of the game to be fed into the neural network
            the state is composed of 5 cards (3 in hand, 1 played card on table, 1 briscola)
            each card is array of size 14, separating one hot encoded number and seed i.e. [number_one_hot, seed_one_hot]
            if there are no cards at a particular location, the array is all zeros.
        '''

        # Reordering the player hand in descending order (high value -> low value)
        #game.reorder_hand(player.id)

        state = np.zeros(self.n_features)
        # add hand to state

        for i, card in enumerate(player.hand):
            number_index = i * 14 + card.number
            state[number_index] = 1
            seed_index = i * 14 + 10 + card.seed
            state[seed_index] = 1
        # add played cards to state
        for i, card in enumerate(game.played_cards):
            number_index = (i + 3) * 14 + card.number
            state[number_index] = 1
            seed_index = (i + 3) * 14 + 10 + card.seed
            state[seed_index] = 1
        # add briscola to state
        number_index = 4 * 14 + game.briscola.number
        state[number_index] = 1
        seed_index = 4 * 14 + 10 + game.briscola.seed
        state[seed_index] = 1
        # add seen cards
        #for card in game.history:
            #card_index = 5 * 14 + card.id
            #state[card_index] = 1


        self.last_state = self.state
        self.state = state
        self.terminal = int(game.check_end_game())


    def select_action(self, available_actions):
        '''Selects an action given the observed state'''

        if self.state is None:
            raise ValueError("DeepAgent.select_action called before observing the state")

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

            #if action != argmax[0]:
                #self.wrong_move = True
                #self.count_wrong_moves += 1

        # store the chosen action
        self.action = action
        return action


    def update(self, reward):
        ''' After receiving a reward the agent has all collected [s, a, r, s_]'''

        '''
        if self.wrong_move:
            # reduce the reward if the agent's last move has been a not allowed move
            self.reward = -10
            self.wrong_move = False
        else:
            self.reward = reward
        '''

        # update last reward
        self.reward = reward

        # update epsilon grediness
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment
            if self.epsilon >= self.epsilon_max:
                self.epsilon = self.epsilon_max
                print("Epsilon max: ", self.epsilon_max, " reached!")

        self.q_learning.learn(self.last_state, self.action, self.reward, self.state, self.terminal)


    def save_model(self, output_dir):
        self.q_learning.save_model(output_dir)

    def load_model(self, saved_model_dir):
        self.q_learning.load_model(saved_model_dir)

    def make_greedy(self):
        self.epsilon_backup = self.epsilon
        self.epsilon = 1.0

    def restore_epsilon(self):
        self.epsilon = self.epsilon_backup

    @staticmethod
    def pad_to_n(state, n):
        target_length = n
        state = np.pad(state, (0, target_length - len(state)), 'constant')
        return state
