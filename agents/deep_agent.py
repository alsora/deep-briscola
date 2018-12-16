import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil

from agents_base.deep_agent import DeepAgent as DeepAgentBase


class DeepAgent(DeepAgentBase):
    ''' Trainable agent which uses a neural network to determine best action'''

    def __init__(self, epsilon=0.85, epsilon_increment=0, epsilon_max = 0.85, discount=0.95, learning_rate = 1e-3):
        self.n_actions = 3
        self.n_features = 70
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon
        self.epsilon_backup = epsilon
        self.epsilon_increment = epsilon_increment
        self.gamma = discount
        # initialize super class with neural network implementation
        super().__init__(self.n_actions, self.n_features, learning_rate)

        self.count_wrong_moves = 0


    def observe(self, game, player, deck):
        ''' create an encoded state representation of the game to be fed into the neural network
            the state is composed of 5 cards (3 in hand, 1 played card on table, 1 briscola)
            each card is array of size 14, separating one hot encoded number and seed i.e. [number_one_hot, seed_one_hot]
        '''

        # the state is array of size 70
        state=np.zeros(70)
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

        self.last_state = self.state
        self.state = state


    def select_action(self, available_actions):
        '''Selects an action given the observed state'''

        if self.state is None:
            raise ValueError("DeepAgent.select_action called before observing the state")

        if np.random.uniform() > self.epsilon:
            # select action randomly with probability (1 - epsilon)
            action = np.random.choice(available_actions)
        else:
            # select action that maximize q value expectation with probability (epsilon)
            states_op = self.session.graph.get_operation_by_name("states").outputs[0]
            argmax_op = self.session.graph.get_operation_by_name("predictions/argmax").outputs[0]
            q_op = self.session.graph.get_operation_by_name("eval_net/q/BiasAdd").outputs[0]

            input_state = np.expand_dims(self.state, axis=0)
            argmax, q = self.session.run([argmax_op, q_op], feed_dict={states_op: input_state})

            q = q[0]
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