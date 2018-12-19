import itertools, time, random
import numpy as np
from enum import Enum

class LoggerLevels(Enum):
    DEBUG = 0
    PVP = 1
    TEST = 2
    TRAIN = 3

class TicTacToeGame:

    def __init__(self, num_players = 2, verbosity=LoggerLevels.TEST):
        self.num_players = num_players
        self.configure_logger(verbosity)


    def configure_logger(self, verbosity):

        if verbosity.value > LoggerLevels.DEBUG.value:
            self.DEBUG_logger = lambda *args: None
        else:
            self.DEBUG_logger = print

        if verbosity.value > LoggerLevels.PVP.value:
            self.PVP_logger = lambda *args: None
        else:
            self.PVP_logger = print

        self.TEST_logger = print
        self.TRAIN_logger = print


    def reset(self):
        self.board = [-1.0] * 9
        self.turn_player = random.randint(0,1)
        self.players_order = self.get_players_order()


    def get_players_order(self):
        ''' compute the clockwise players order starting from the current turn player'''
        players_order = [ i % self.num_players for i in range(self.turn_player, self.turn_player + self.num_players)]
        return players_order


    def play_step(self, action, player_id):
        ''' a player executes a chosen action'''

        self.PVP_logger("Player ", player_id, " choose action ", action)

        if self.is_space_free(self.board, action):
            self.write_cell(self.board, action, player_id)
        else:
            raise ValueError("Trying to write non free cell!!!!")

    def evaluate_step(self, player_id):

        win = False
        draw = False

        if self.check_winner(self.board, player_id):
            win = True
        elif self.check_board_full(self.board):
            draw = True

        return win, draw


    @staticmethod
    def write_cell(board, index, marker):
        board[index] = marker

    @staticmethod
    def get_player_actions(board, player_id):
        ''' get list of available actions for a player'''
        return [x for x in range(9) if TicTacToeGame.is_space_free(board, x)]

    @staticmethod
    def is_space_free(board, index):
        "checks for free space of the board"
        return board[index] == -1.0

    @staticmethod
    def check_winner(board, marker):
        winning_combos = (
            [6, 7, 8], [3, 4, 5], [0, 1, 2], [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],)

        for combo in winning_combos:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]] == marker):
                return True
        return False

    @staticmethod
    def check_board_full(board):
        "checks if the board is full"
        for i in range(9):
            if TicTacToeGame.is_space_free(board, i):
                return False
        return True


    @staticmethod
    def print_board(board):
        board = [int(i) for i in board]
        board_matrix = np.reshape(board, (3, 3))
        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
            for row in board_matrix]))



