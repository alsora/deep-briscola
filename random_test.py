import tensorflow as tf

from agents.random_agent import RandomAgent
import environment as brisc



def main(argv=None):
    # Initializing the environment
    game = brisc.BriscolaGame(2,verbosity=brisc.LoggerLevels.DEBUG)
    deck = game.deck

    # Initialize agents
    agents = []
    agents.append(RandomAgent())
    agents.append(RandomAgent())

    # First reset of the environment
    game.reset()
    keep_playing = True

    while keep_playing:

        players_order = game.get_players_order()
        for player_id in players_order:

            player = game.players[player_id]
            agent = agents[player_id]

            agent.observe(game, player, deck)
            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

        winner_player_id, points = game.evaluate_step()

        keep_playing = game.draw_step()


    game_winner_id, winner_points = game.end_game()



if __name__ == '__main__':
    tf.app.run()
