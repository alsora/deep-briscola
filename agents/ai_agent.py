import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import math
import numpy as np
import environment as tictac

class AIAgent:

    def __init__(self):
        pass


    def observe(self, game, player_id):
        ''' store information about the state of the game to be used in the decisional process'''
        self.board = game.board
        self.id = player_id


    def select_action(self, available_actions):
        '''Selects an action given the observed state'''

        if len(available_actions) == 9:
            action = np.random.choice(available_actions)
        else:
            action, _ = self.minmax_predict(self.id, copy.deepcopy(self.board))

        self.action = action
        return action



    def minmax_predict(self, turn_player, board):
        ''' recursive implementation of minmax tree search.
            it never loses as long as the opponent does perfect moves'''

        # check end game and return empty action, end game reward
        if tictac.TicTacToeGame.check_board_full(board):
            if tictac.TicTacToeGame.check_winner(board, self.id):
                return -1, 1
            elif tictac.TicTacToeGame.check_winner(board, 1 - self.id):
                return -1, -1
            else:
                return -1, 0

        # set initial values for best action and best score
        if turn_player == self.id:
            best_action = -1
            best_score = -math.inf
        else:
            best_action = -1
            best_score = +math.inf

        available_actions = tictac.TicTacToeGame.get_player_actions(board, turn_player)
        for action in available_actions:
            b = copy.deepcopy(board)
            tictac.TicTacToeGame.write_cell(b, action, turn_player)
            mm_action, mm_score, = self.minmax_predict(1 - turn_player, b)
            mm_action = action

            if turn_player == self.id:
                if mm_score > best_score:
                    best_score = mm_score # max value
                    best_action = mm_action
            else:
                if mm_score < best_score:
                    best_score = mm_score # min value
                    best_action = mm_action

        return best_action, best_score


    def update(self, reward):
        pass


    def make_greedy(self):
        pass


    def restore_epsilon(self):
        pass