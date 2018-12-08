import itertools, time, random
import numpy as np

from errors_classes import CardsFinished,InvalidAction,PlayerIdError, DrawingProblem

# Immutable variables
points = [11,0,10,0,0,0,0,2,3,4]
strengths = [10,1,9,2,3,4,5,6,7,8]
seeds = ['Spade','Coppe','Denari','Bastoni']
names = ['Asso', 'Due', 'Tre', 'Quattro', 'Cinque', 'Sei', 'Sette', 'Fante', 'Cavallo', 'Re']

class Card:

    def __init__(self):
        self.id = -1
        self.name = None
        self.seed = -1
        self.strength = -1
        self.points = -1


class BriscolaDeck:
    global names, seeds

    def __init__(self):
        self.create_deck()
        self.reset()

    def create_deck(self):
        self.deck = []
        id = 0
        for s, seed in enumerate(seeds):
            for n, name in enumerate(names):
                card = Card()
                card.id = id
                card.name = name + ' di ' + seed
                card.seed = s
                card.strength = strengths[n]
                card.points = points[n]
                self.deck.append(card)
                id += 1

    def place_briscola(self, briscola):
        self.briscola = briscola

    def reset(self):
        self.current_deck = self.deck.copy()
        self.briscola = None
        self.last_turn_done = False

    def get_deck_size(self):
        return len(self.deck)

    def get_current_deck_size(self):
        return len(self.current_deck)

    def get_card_name(self, id):
        return self.deck[id].name

    def get_card_names(self, ids):
        return[self.deck[id].name for id in ids ]

    def get_card(self, id):
        return self.deck[id]

    def get_card_one_hot(self, id):
        #one_hot_vector = np.zeros(self.get_deck_size())
        #one_hot_vector[id] = 1
        one_hot_vector = [1 if i == id else 0 for i in list(range(self.get_deck_size()))]
        return one_hot_vector

    def get_cards_one_hot(self, ids):
        one_hot_vector = [1 if id in ids else 0 for id in list(range(self.get_deck_size()))]
        return one_hot_vector

    def draw_card(self):
        """
        tool that allows drawing card
        """

        drawn_card = None

        if len(self.current_deck) == 0:
            if self.briscola:
                print ("DRAW -----> BRISCOLA")
                drawn_card = self.briscola
                self.briscola = None
            else:
                print ("TRY TO DRAW BUT DECK EMPTY")
                return None
        else:
            available_cards_id = list(range(0, self.get_current_deck_size()))
            drawn_card_id = np.random.choice(available_cards_id)
            drawn_card = self.current_deck[drawn_card_id]
            del self.current_deck[drawn_card_id]

        return drawn_card



class BriscolaPlayer:

    action_space = ['first_card','second_card','third_card']

    def __init__(self, _id):
        self.id = _id
        self.reset()

    def reset(self):
        self.last_action = ''
        self.last_thee_turn = False  # TODO: do this in a better way pls
        self.game_going = True

        self.hand = []
        self.points = 0

    def draw(self, deck):
        """
            the player draws a card from the deck
        """
        new_card = deck.draw_card()

        if new_card is None and len(self.hand) is 0:
            return False

        if new_card is not None:
            self.hand.append(new_card)

        if len(self.hand) > 3:
            raise DrawingProblem

        return True


    def play_card(self, hand_index):

        try:
            card = self.hand[hand_index]
            del self.hand[hand_index]
            return card.id
        except:
            print("PLAY CARD EXCEPTION----------_>")
            return None


    def get_player_state(self):
        return self.hand


class BriscolaGame:

    def __init__(self, summary_turn):
          self.summary_turn = summary_turn
          self.deck = BriscolaDeck()

    def reset(self):
        self.deck.reset()
        self.history = []
        self.played_cards = []

        # Initilize the players
        self.players = [BriscolaPlayer(i) for i in range(2)]
        self.turn_player = random.randint(1,2)

        self.players_order = self.get_players_order()

        self.turn_state = {
            'played_cards': self.played_cards,
            'players_order': self.players_order
        }

        # Initialize the briscola
        self.briscola = self.deck.draw_card()
        self.briscola_seed = self.briscola.seed

        self.deck.place_briscola(self.briscola)

        for _ in range(0,3):
            for i in self.players_order:
                self.players[i].draw(self.deck)

        self.print_summary()


    def get_player_actions(self, player_id):
        player = self.players[player_id]
        return list(range(len(player.hand)))

    def get_turn_state(self):
        return self.turn_state

    def get_players_order(self):

        players_order = [ i % len(self.players) for i in range(self.turn_player, self.turn_player + len(self.players))]
        return players_order

    def draw_step(self):

        for player_id in self.players_order:
            player = self.players[player_id]

            ret = player.draw(self.deck)

            if not ret:
                return False

        return True


    def play_step(self, action, player_id):

        player = self.players[player_id]
        print ("PLAYER: ", player_id, "HAND: ", [card.name for card in player.hand])
        print ("ACTION---->", action)
        card_id = player.play_card(action)
        if card_id is None:
            print ("PLAY_CARD IS NONE!!!!!------------------")
        card = self.deck.get_card(card_id)
        print("Played card: ", card.name)

        self.played_cards.append(card)
        #self.turn_state['played_cards'].append(card)


    def evaluate_step(self):

        '''
        winner_player_id = self.players_order[0]
        strongest_card = self.played_cards[0]

        for ordered_player_id, card in enumerate(self.played_cards[1:]):
            pair_winner = self.scoring(strongest_card, card)
            if pair_winner is 1:
                winner_player_id = self.players_order[ordered_player_id + 1]
                strongest_card = card
        print("ordered_winner->", ordered_player_id , " + 1 = ", ordered_player_id + 1)

        '''

        print ("eVAluate cards: ", [card.name for card in self.played_cards])

        ordered_winner_id, strongest_card = self.get_strongest_card(self.briscola_seed, self.played_cards)
        winner_player_id = self.players_order[ordered_winner_id]

        points = sum([card.points for card in self.played_cards])
        print ("players order ", self.players_order)
        winner_player = self.players[winner_player_id]

        print ("Winner player ", winner_player_id, " with ", strongest_card.name)
        print ("Points: ", points)

        self.update_game(winner_player, points)

        return winner_player_id, points

    @staticmethod
    def get_strongest_card(briscola_seed, cards):

        ordered_winner_id = 0
        strongest_card = cards[0]

        for ordered_player_id, card in enumerate(cards[1:]):
            ordered_player_id += 1 # adjustment since we are starting from firsr element
            pair_winner = BriscolaGame.scoring(briscola_seed, strongest_card, card)
            if pair_winner is 1:
                ordered_winner_id = ordered_player_id
                strongest_card = card

        return ordered_winner_id, strongest_card

    @staticmethod
    def scoring(briscola_seed, card_0, card_1):

        card_0_seed = card_0.seed
        card_1_seed = card_1.seed

        winner = 0

        if briscola_seed is not card_0_seed and briscola_seed is card_1_seed:
            winner = 1
        elif card_0_seed is card_1_seed:
            winner = 1 if card_1.strength > card_0.strength else 0
        # if different seeds and none of them is briscola, first wins

        return winner



    def end_game(self):

        print ("End game!")

        winner_player_id = -1
        winner_points = -1

        for player in self.players:
            print("Player ", player.id, " made ", player.points, " points")
            if player.points > winner_points:
                winner_player_id = player.id
                winner_points = player.points

        return winner_player_id, winner_points


    def update_game(self, winner_player, points):

        winner_player_id = winner_player.id
        winner_player.points += points

        self.turn_state['winner_player_id'] = winner_player_id
        self.turn_state['points'] = points

        self.history.append(self.turn_state)

        self.played_cards = []
        self.turn_player = winner_player_id
        self.players_order= self.get_players_order()

        self.turn_state = {
            'played_cards': self.played_cards,
            'players_order': self.players_order
        }



    def print_summary(self):

        for player in self.players:
            self.print_hand(player)

        self.print_briscola()
        #self.print_last_action()


    def print_hand(self, player):

        text = ''
        if len(player.hand) > 0:
            text += 'Player ' + str(player.id) + ' has hand: '
            for i, card in enumerate(player.hand):
                if i != 0:
                    text += ', '
                text += card.name

        print(text)

    def print_briscola(self):

        briscola_name = self.briscola.name
        print('The briscola is: ', briscola_name)


