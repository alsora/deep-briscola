import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil

from agents_base.deep_agent import DeepAgent as DeepAgentBase


class DeepAgent(DeepAgentBase):

    def __init__(self):
        n_actions = 3
        n_features = 70
        super().__init__(n_actions, n_features)

        self.count_wrong_moves = 0


    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        #self.observed_state['hand_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['hand']])
        self.observed_state['turn_state'] = game.get_turn_state()
        self.observed_state['briscola'] = game.briscola
        self.observed_state['briscola_seed'] = game.briscola_seed
        #self.observed_state['briscola_one_hot'] = deck.get_card_one_hot(game.briscola.id)
        #self.observed_state['played_cards_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['turn_state']['played_cards']])

        #hand_one_hot = np.array(self.observed_state['hand_one_hot'])
        #briscola_one_hot = np.array(self.observed_state['briscola_one_hot'])
        #played_cards_one_hot = np.array(self.observed_state['played_cards_one_hot'])

        # (1,70) each card is (1,14) separating id and seed
        state=np.array([])
        for i, card in enumerate(self.observed_state['hand']):
            id_one_hot = np.zeros(10)
            seed_one_hot = np.zeros(4)
            id_one_hot[card.number] = 1
            seed_one_hot[card.seed] = 1
            state = np.concatenate((state, np.concatenate((id_one_hot, seed_one_hot), axis=0)), axis=0)
        target_length = 14 * 3
        state = np.pad(state, (0, target_length - len(state)), 'constant')
        for i, card in enumerate(self.observed_state['turn_state']['played_cards']):
            id_one_hot = np.zeros(10)
            seed_one_hot = np.zeros(4)
            id_one_hot[card.number] = 1
            seed_one_hot[card.seed] = 1
            state = np.concatenate((state, np.concatenate((id_one_hot, seed_one_hot), axis=0)), axis=0)
        target_length = 14 * 4
        state = np.pad(state, (0, target_length - len(state)), 'constant')
        briscola_id_one_hot = np.zeros(10)
        briscola_seed_one_hot = np.zeros(4)
        briscola_id_one_hot[self.observed_state['briscola'].number] = 1
        briscola_seed_one_hot[self.observed_state['briscola'].seed] = 1
        state = np.concatenate((state, np.concatenate((briscola_id_one_hot, briscola_seed_one_hot), axis=0)), axis=0)

        '''
        # (1,5) compact input
        #state = np.array([game.briscola.id])
        state = np.array([game.briscola_seed])
        for card in self.observed_state['hand']:
            state = np.append(state, np.array([card.id]))
        target_length = 4
        state = np.pad(state, (0, target_length - len(state)), 'constant', constant_values=(99,99))
        for card in self.observed_state['turn_state']['played_cards']:
            state = np.append(state, np.array([card.id]))
        target_length = 5
        state = np.pad(state, (0, target_length - len(state)), 'constant', constant_values=(99,99))
        '''
        '''
        # (1, 40) state: PROBLEM: briscola is overwritten once drawn
        state = 10*briscola_one_hot -1*played_cards_one_hot
        for i, card in enumerate(self.observed_state['hand']):
            state += (i+1) * deck.get_card_one_hot(card.id)
        '''
        '''
        # (1, 120) state: 40 hand + 40 played + 40 briscola
        for i, card in enumerate(self.observed_state['hand']):
            state += (i+1) * deck.get_card_one_hot(card.id)
        state = np.concatenate((state, played_cards_one_hot), axis=0)
        state = np.concatenate((state, briscola_one_hot), axis=0)
        '''
        '''
        # (1, 200) state (hand is not merged) PROBLEM: extremely sparse
        state = np.array([])
        for card in self.observed_state['hand']:
            state = np.concatenate((state, deck.get_card_one_hot(card.id)), axis=0)
        target_length = 40 * 3
        state = np.pad(state, (0, target_length - len(state)), 'constant')
        state = np.concatenate((state, played_cards_one_hot), axis=0)
        state = np.concatenate((state, briscola_one_hot), axis=0)
        '''


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

        self.action = action
        return action