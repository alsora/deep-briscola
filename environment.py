import itertools, time, random
import numpy as np
from enum import Enum

class LoggerLevels(Enum):
    DEBUG = 0
    PVP = 1
    TEST = 2
    TRAIN = 3

class BriscolaCard:

    def __init__(self):
        self.id = -1        # index of the one-hot encoded card [0, len(deck)]
        self.name = ''      # name to display
        self.seed = -1      # seed of the card [0, 3]
        self.number = -1    # number of the card [0, 9]
        self.strength = -1  # ordered number of the card [0, 9]
        self.points = -1    # points value of the card [0, 11]




class BriscolaDeck:

    def __init__(self):
        self.create_decklist()
        self.reset()


    def create_decklist(self):
        ''' Create all the BriscolaCard and add them to deck'''
        points = [11,0,10,0,0,0,0,2,3,4]
        strengths = [9,0,8,1,2,3,4,5,6,7]
        seeds = ['Spade','Coppe','Denari','Bastoni']
        names = ['Asso', 'Due', 'Tre', 'Quattro', 'Cinque', 'Sei', 'Sette', 'Fante', 'Cavallo', 'Re']

        self.deck = []
        id = 0
        for s, seed in enumerate(seeds):
            for n, name in enumerate(names):
                card = BriscolaCard()
                card.id = id
                card.name = name + ' di ' + seed
                card.seed = s
                card.number = n
                card.strength = strengths[n]
                card.points = points[n]
                self.deck.append(card)
                id += 1


    def reset(self):
        ''' Prepare the deck for a new game'''
        self.briscola = None
        self.end_deck = False
        self.current_deck = self.deck.copy()
        self.shuffle()


    def shuffle(self):
        ''' Shuffle the deck'''
        random.shuffle(self.current_deck)


    def place_briscola(self, briscola):
        ''' Set a card as briscola and allows to draw it after last card of the deck'''
        if self.briscola is not None:
            raise ValueError("Trying BriscolaDeck.place_briscola, but BriscolaDeck.briscola is not None")
        self.briscola = briscola


    def draw_card(self):
        ''' If deck is not empty, then draw a card, else return the briscola or nothing'''
        if self.current_deck:
            drawn_card =  self.current_deck.pop()
        else:
            drawn_card = self.briscola
            self.briscola = None
            self.end_deck = True

        return drawn_card


    def get_deck_size(self):
        ''' Size of the full deck'''
        return len(self.deck)


    def get_current_deck_size(self):
        '''Size of the current deck'''
        current_deck_size = len(self.current_deck)
        current_deck_size += 1 if self.briscola else 0
        return current_deck_size

    def card_value(self,card_br):
        card, br = card_br
        br_seed = br.seed
        card_value = (card.seed == br_seed) * 10 + card.points
        return card_value



class BriscolaPlayer:

    def __init__(self, _id):
        self.id = _id
        self.reset()


    def reset(self):
        self.hand = []
        self.points = 0


    def draw(self, deck):
        ''' Try to draw a card from the deck'''

        new_card = deck.draw_card()
        if new_card is not None:
            self.hand.append(new_card)

        if len(self.hand) > 3:
            raise ValueError("player.draw caused the player to have more than 3 cards in hand!")


    def play_card(self, hand_index):
        ''' Try to play a card from the hand and return the chosen card or None if invalid index'''

        try:
            card = self.hand[hand_index]
            del self.hand[hand_index]
            return card
        except:
            raise ValueError("player.play_card called with invalid hand_index!")
            return None




class BriscolaGame:

    def __init__(self, num_players = 2, verbosity=LoggerLevels.TEST):
        self.num_players = num_players
        self.deck = BriscolaDeck()
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
        ''' starts a new game'''
        self.deck.reset()
        self.history = []
        self.played_cards = []

        # Initilize the players
        self.players = [BriscolaPlayer(i) for i in range(self.num_players)]
        self.turn_player = random.randint(0, self.num_players - 1)
        self.players_order = self.get_players_order()

        # Initialize the briscola
        self.briscola = self.deck.draw_card()
        self.deck.place_briscola(self.briscola)

        for _ in range(0,3):
            for i in self.players_order:
                self.players[i].draw(self.deck)


    def reorder_hand(self, player_id):
        ''' reorders the cards in a player hand from strongest to weakest,
            taking briscola seed into account
        '''
        player = self.players[player_id]

        # bubble sort algorithm using scoring() as comparator
        for passnum in range(len(player.hand)-1,0,-1):
            for i in range(passnum):
                if scoring(self.briscola.seed, player.hand[i], player.hand[i+1], keep_order=False):
                    temp = player.hand[i]
                    player.hand[i] = player.hand[i+1]
                    player.hand[i+1] = temp


    def get_player_actions(self, player_id):
        ''' get list of available actions for a player'''
        player = self.players[player_id]
        return list(range(len(player.hand)))


    def get_players_order(self):
        ''' compute the clockwise players order starting from the current turn player'''
        players_order = [ i % self.num_players for i in range(self.turn_player, self.turn_player + self.num_players)]
        return players_order


    def draw_step(self):
        ''' each player, in order, tries to draw a card'''
        self.PVP_logger("----------- NEW TURN -----------")

        for player_id in self.players_order:
            player = self.players[player_id]

            player.draw(self.deck)



    def play_step(self, action, player_id):
        ''' a player executes a chosen action'''

        player = self.players[player_id]

        self.DEBUG_logger("Player ", player_id, " hand: ", [card.name for card in player.hand])
        self.DEBUG_logger("Player ", player_id, " choose action ", action)

        card = player.play_card(action)
        if card is None:
            raise ValueError("player.play_card failed!")

        self.PVP_logger("Player ", player_id, " played ", card.name)

        self.played_cards.append(card)
        self.history.append(card)


    def get_rewards_from_step(self):
        ''' compute rewards for each player according to the just played cards'''

        winner_player_id, points = self.evaluate_step()

        rewards = []
        for player_id in self.get_players_order():
            player = self.players[player_id]

            reward = points if player_id is winner_player_id else -points
            #reward = points if player_id is winner_player_id else 0

            rewards.append(reward)

        return rewards


    def evaluate_step(self):
        ''' look at played cards and decide which player won the hand'''

        ordered_winner_id, strongest_card = get_strongest_card(self.briscola.seed, self.played_cards)
        winner_player_id = self.players_order[ordered_winner_id]

        points = sum([card.points for card in self.played_cards])
        winner_player = self.players[winner_player_id]

        self.update_game(winner_player, points)

        self.PVP_logger("Player ", winner_player_id, " wins ", points, " points with ", strongest_card.name)

        return winner_player_id, points


    def check_end_game(self):
        ''' check if the game is ended'''
        end_deck = self.deck.end_deck
        player_has_cards = False
        for player in self.players:
            if player.hand:
                player_has_cards = True
                break

        return (end_deck and not player_has_cards)


    def get_winner(self):
        ''' returns the player with most_points'''
        winner_player_id = -1
        winner_points = -1

        for player in self.players:
            if player.points > winner_points:
                winner_player_id = player.id
                winner_points = player.points

        return winner_player_id, winner_points


    def end_game(self):
        ''' returns id of the winner of the game'''
        if not self.check_end_game():
            raise ValueError('Calling BriscolaGame.end_game when the game has not ended!')

        winner_player_id, winner_points = self.get_winner()

        self.PVP_logger("Player ", winner_player_id, " wins with ", winner_points, " points!!")

        return winner_player_id, winner_points


    def update_game(self, winner_player, points):

        winner_player_id = winner_player.id
        winner_player.points += points

        self.played_cards = []
        self.turn_player = winner_player_id
        self.players_order = self.get_players_order()




def get_strongest_card(briscola_seed, cards):
    ''' Get the strongest card in the provided set'''
    ordered_winner_id = 0
    strongest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1 # adjustment since we are starting from firsr element
        pair_winner = scoring(briscola_seed, strongest_card, card)
        if pair_winner is 1:
            ordered_winner_id = ordered_id
            strongest_card = card

    return ordered_winner_id, strongest_card


def get_weakest_card(briscola_seed, cards):
    ''' Get the weakest card in the provided set'''
    ordered_loser_id = 0
    weakest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1 # adjustment since we are starting from firsr element
        pair_winner = scoring(briscola_seed, weakest_card, card, keep_order=False)
        if pair_winner is 0:
            ordered_loser_id = ordered_id
            weakest_card = card

    return ordered_loser_id, weakest_card


def scoring(briscola_seed, card_0, card_1, keep_order=True):
    ''' compare a pair of cards and decide who wins.
        keep_order variable decides wether the first played card has a priority
    '''

    if briscola_seed is not card_0.seed and briscola_seed is card_1.seed:
        winner = 1
    elif briscola_seed is card_0.seed and briscola_seed is not card_1.seed:
        winner = 0
    elif card_0.seed is card_1.seed:
        winner = 1 if card_1.strength > card_0.strength else 0
    else:
        # if different seeds and none of them is briscola, first wins
        winner = 0 if keep_order or card_0.points > card_1.points else 1

    return winner



def play_episode(game, agents, train=True):

    game.reset()
    while not game.check_end_game():

        # action step
        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]
            # agent observes state before acting
            agent.observe(game, player, game.deck)
            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

        rewards = game.get_rewards_from_step()
        # update agents if training mode
        if train:
            for i, player_id in enumerate(players_order):
                player = game.players[player_id]
                agent = agents[player_id]
                # agent observes new state after acting
                agent.observe(game, player, game.deck)

                reward = rewards[i]
                agent.update(reward)

        # update the environment
        game.draw_step()

    return game.end_game()