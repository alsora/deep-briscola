import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import environment as brisc

class AIAgent:

    def __init__(self):
        self.name = 'AIAgent'


    def observe(self, game, player):
        ''' store information about the state of the game to be used in the decisional process'''
        self.hand = player.hand
        self.points = player.points
        self.played_cards = game.played_cards
        self.briscola_seed = game.briscola.seed


    def select_action(self, actions):

        # count how many points are present on table
        points_on_table =  0
        for played_card in self.played_cards:
            points_on_table += played_card.points

        if points_on_table:
            # there is at least 1 card on table and it's worth some points
            win_actions = []
            points = []
            for action_index, card in enumerate(self.hand):
                for played_card in self.played_cards:
                    winner = brisc.scoring(self.briscola_seed, played_card, card)
                    if winner:
                        win_actions.append(action_index)
                        points.append(card.points)
            if win_actions:
                # if I can win this hand
                # my cards which can win the hand, sorted from the most points I can earn from playing it
                sorted_win_actions = [x for _,x in sorted(zip(points, win_actions), reverse=True)]
                best_action = sorted_win_actions[0]
                best_card = self.hand[best_action]

                if self.points + points_on_table + best_card.points > 60:
                    # if winning this hand I can win the game, do it
                    return best_action

                # if I can win without using a briscola, do it
                for win_action in sorted_win_actions:
                    if self.hand[win_action].seed != self.briscola_seed:
                        return win_action

                # if I am here, all the cards which allow me to win the hand are briscola
                if len(win_actions) == 1:
                    # if I have only one card for winning the hand (which is a briscola)
                    if points_on_table >= 10:
                        # if there are many points on the table, always play the briscola
                        return best_action

                    # if there are not many points on the table and the other losing cards in hand are carichi, play the briscola
                    # if I have a low points, non-briscola alternative to lose the hand, play the alternative
                    lose_action = -1
                    lose_points = 10
                    for action_index, card in enumerate(self.hand):
                        if action_index not in win_actions and card.points < lose_points and card.seed != self.briscola_seed:
                            lose_action = action_index
                            lose_points = card.points
                    if lose_action is -1:
                        # I only have carichi losing cards and 1 winning briscola in hand, play the briscola
                        return best_action
                    elif best_card.points >= 3:
                        # If my only winning briscola is cavallo or more lose the hand, since there are not many points on the table
                        return lose_action
                    elif lose_points == 4:
                        # I have in hand: 2 re 1 briscola up to fante or 1 carico 1 re 1 briscola up to fante
                        if points_on_table >=3:
                            # if there are at least 3 points on table, play briscola up to fante
                            return best_action
                        else:
                            return lose_action
                    else:
                        # I can play a less than 4 points, non-briscola, losing alternative, play it
                        return lose_action
                else:
                    # I have more than one briscola for winning the hand, play the weakest
                    win_cards = [self.hand[i] for i in win_actions]
                    weakest_win_index, weakest_win_card = brisc.get_weakest_card(self.briscola_seed, win_cards)
                    return win_actions[weakest_win_index]

        # if I am here I can't win the hand or there are no points on table (there may be no card at all)
        # find weakest card (it may be a carico if other cards in hand are briscola)
        weakest_index, weakest_card = brisc.get_weakest_card(self.briscola_seed, self.hand)
        if weakest_card.points > 4:
            # I would rather throw a small briscola than a carico
            low_points_sorted_cards = sorted(self.hand, key=lambda card: card.strength)
            return self.hand.index(low_points_sorted_cards[0])
        else:
            return weakest_index


    def update(self, reward):
        pass


    def make_greedy(self):
        pass


    def restore_epsilon(self):
        pass