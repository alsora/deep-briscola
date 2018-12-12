import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil

from agents_base.deep_agent import DeepAgent as DeepAgentBase


class DeepAgent(DeepAgentBase):

    def __init__(self, epsilon=0.85, epsilon_increment=0, discount=0.95):
        self.n_actions = 3
        self.n_features = 70
        self.epsilon_max = 0.99
        self.epsilon = epsilon
        if self.epsilon > self.epsilon_max:
            self.epsilon = self.epsilon_max
        self.epsilon_increment = epsilon_increment
        self.gamma = discount # reward discount factor
        super().__init__(self.n_actions, self.n_features)

        self.count_wrong_moves = 0


    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        self.observed_state['played_cards'] = game.played_cards
        self.observed_state['briscola'] = game.briscola
        self.observed_state['briscola_seed'] = game.briscola_seed

        # (1,70) each card is (1,14) separating id and seed
        state=np.array([])
        for i, card in enumerate(self.observed_state['hand']):
            id_one_hot = np.zeros(10)
            seed_one_hot = np.zeros(4)
            id_one_hot[card.number] = 1
            seed_one_hot[card.seed] = 1
            state = np.concatenate((state, np.concatenate((id_one_hot, seed_one_hot), axis=0)), axis=0)
        state = self.pad_to_n(state, 14 * 3)
        for i, card in enumerate(self.observed_state['played_cards']):
            id_one_hot = np.zeros(10)
            seed_one_hot = np.zeros(4)
            id_one_hot[card.number] = 1
            seed_one_hot[card.seed] = 1
            state = np.concatenate((state, np.concatenate((id_one_hot, seed_one_hot), axis=0)), axis=0)
        state = self.pad_to_n(state, 14 * 4)
        briscola_id_one_hot = np.zeros(10)
        briscola_seed_one_hot = np.zeros(4)
        briscola_id_one_hot[self.observed_state['briscola'].number] = 1
        briscola_seed_one_hot[self.observed_state['briscola'].seed] = 1
        state = np.concatenate((state, np.concatenate((briscola_id_one_hot, briscola_seed_one_hot), axis=0)), axis=0)

        self.last_state = self.state
        self.state = state



    def select_action(self, actions):

        if np.random.uniform() > self.epsilon:
            action = np.random.choice(actions)
            self.action = action
            return action

        states_op = self.session.graph.get_operation_by_name("states").outputs[0]
        predictions_op = self.session.graph.get_operation_by_name("predictions/argmax").outputs[0]

        input_state = np.expand_dims(self.state, axis=0)
        predictions, q_eval = self.session.run([predictions_op, self.q_eval], feed_dict={states_op: input_state})

        action = predictions[0]

        if action not in actions:
            #print ("Selected invalid action!!!")
            self.wrong_move = True
            self.count_wrong_moves += 1
            action = np.random.choice(actions)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.action = action
        return action

    @staticmethod
    def pad_to_n(state, n):
        target_length = n
        state = np.pad(state, (0, target_length - len(state)), 'constant')
        return state