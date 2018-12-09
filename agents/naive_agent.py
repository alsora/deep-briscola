import numpy as np
import environment as brisc

class NaiveAgent:

    def __init__(self):
        self.observed_state = {}

    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        self.observed_state['hand_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['hand']])
        self.observed_state['turn_state'] = game.get_turn_state()
        self.observed_state['briscola_seed'] = game.briscola_seed
        self.observed_state['briscola_one_hot'] = deck.get_card_one_hot(game.briscola.id)
        self.observed_state['played_cards_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['turn_state']['played_cards']])


    def select_action(self, actions):

        played_cards = self.observed_state['turn_state']['played_cards']
        hand =  self.observed_state['hand']
        briscola_seed = self.observed_state['briscola_seed']

        possible_winners = hand + played_cards
        winner_index, strongest_card = brisc.BriscolaGame.get_strongest_card(briscola_seed, possible_winners)

        if winner_index < len(hand) and len(played_cards) > 0:
            action = winner_index
        else:
            action = np.random.choice(actions)

        return action


    def update(self, reward, new_state):
        pass