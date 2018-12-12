import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import environment as brisc

class AIAgent:

    def __init__(self):
        self.observed_state = {}

    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        self.observed_state['played_cards'] = game.played_cards
        self.observed_state['briscola_seed'] = game.briscola_seed


    def select_action(self, actions):

        played_cards = self.observed_state['played_cards']
        hand =  self.observed_state['hand']
        briscola_seed = self.observed_state['briscola_seed']

        points_on_table =  0
        for played_card in played_cards:
            points_on_table += played_card.points
        # enter if there is at least 1 card on the table and it's worth some points
        if points_on_table:
            win_actions = []
            points = []
            for action_index, card in enumerate(hand):
                for played_card in played_cards:
                    winner = brisc.BriscolaGame.scoring(briscola_seed, played_card, card)
                    if winner:
                        win_actions.append(action_index)
                        points.append(card.points)
            # if I can win this hand
            if win_actions:
                # my cards which can win the hand, sorted from the most points I can earn from playing it
                sorted_win_actions = [x for _,x in sorted(zip(points, win_actions), reverse=True)]
                best_action = sorted_win_actions[0]
                best_card = hand[best_action]
                # if I can win without using a briscola, do it
                for win_action in sorted_win_actions:
                    if hand[win_action].seed != briscola_seed:
                        return win_action
                # if I can only win playing a briscola (best_action is briscola, already checked before)
                if len(win_actions) == 1:
                    # if there are many points on the table, play the briscola
                    if points_on_table >= 10:
                        return best_action
                    # if there are not many points on the table and the other losing cards in hand are carichi, play the briscola
                    # if I have a low points, non-briscola alternative to lose the hand, play the alternative
                    lose_action = -1
                    lose_points = 10
                    for action_index, card in enumerate(hand):
                        if action_index not in win_actions and card.points < lose_points and card.seed != briscola_seed:
                            lose_action = action_index
                            lose_points = card.points
                    # I only have carichi losing cards and 1 winning briscola in hand, play the briscola
                    if lose_action is -1:
                        return best_action
                    # If my only winning briscola is cavallo or more lose the hand, since there are not many points on the table
                    if best_card.points >= 3:
                        return lose_action
                    # I have 2 re 1 briscola up to fante or 1 carico 1 re 1 briscola up to fante
                    if lose_points == 4:
                        # if there are at least 3 points on table, play briscola up to fante
                        if points_on_table >=3:
                            return best_action
                        else:
                            return lose_action
                    # I can play a less than 4 points, non-briscola, losing alternative, play it
                    else:
                        return lose_action
                # I have more than one briscola for winning the hand, play the weakest
                else:
                    win_cards = [hand[i] for i in win_actions]
                    weakest_win_index, weakest_win_card = brisc.BriscolaGame.get_weakest_card(briscola_seed, win_cards)
                    return win_actions[weakest_win_index]
        # if I am here I can't win the hand or there are no points on table (there may be no card at all)
        weakest_index, weakest_card = brisc.BriscolaGame.get_weakest_card(briscola_seed, hand)
        # i would rather throw a small briscola than a carico
        if weakest_card.points > 4:
            low_points_sorted_cards = sorted(hand, key=lambda card: card.strength)
            return hand.index(low_points_sorted_cards[0])
        else:
            return weakest_index


    def update(self, reward):
        pass