import tensorflow as tf

from agents_base.random_agent import RandomAgent
from agents.deep_agent import DeepAgent
import environment as brisc



# Parameters
# ==================================================

# Model directory
tf.flags.DEFINE_string("model_dir", "", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs")

# Evaluation parameters
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate model on dev set after this many steps")

FLAGS = tf.flags.FLAGS

def test(game, agents):

    deck = game.deck
    total_wins = [0, 0]
    total_points = [0, 0]

    for _ in range(FLAGS.num_evaluations):

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

        total_wins[game_winner_id] += 1
        total_points[game_winner_id] += winner_points
        total_points[1 - game_winner_id] += (120 - winner_points)


    victory_rate = (total_wins[0]/float(FLAGS.num_evaluations))*100
    average_points = float(total_points[0])/float(FLAGS.num_evaluations)
    print("DeepAgent wins ", victory_rate, "% with average points ", average_points)

    return victory_rate



def main(argv=None):

    # Initializing the environment
    game = brisc.BriscolaGame(verbosity=brisc.LoggerLevels.TRAIN)
    deck = game.deck

    # Initialize agents
    agents = []
    agents.append(DeepAgent())
    agents.append(RandomAgent())

    best_winning_ratio = -1

    for epoch in range(1, FLAGS.num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')
        game.reset()
        keep_playing = True

        while keep_playing:

            # action step
            players_order = game.get_players_order()
            for player_id in players_order:

                player = game.players[player_id]
                agent = agents[player_id]
                # agent observes state before acting
                agent.observe(game, player, deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)


            winner_player_id, points = game.evaluate_step()

            # update agents
            for player_id in players_order:
                player = game.players[player_id]
                agent = agents[player_id]
                # agent observes new state after acting
                agent.observe(game, player, deck)

                reward = points if player_id is winner_player_id else -points

                '''
                # compute reward function for this player
                if player_id is winner_player_id:
                    reward = points
                elif points >= 10:
                    reward = -2
                else:
                    reward = 0
                '''
                agent.update(reward)

            # update the environment
            keep_playing = game.draw_step()


        game_winner_id, winner_points = game.end_game()
        '''
        # update agents
        for player_id in players_order:
            player = game.players[player_id]
            agent = agents[player_id]

            # compute reward function for this player
            if player_id is game_winner_id:
                reward = 40
            else:
                reward = -20

            agent.update(reward)
        '''

        if epoch % FLAGS.evaluate_every == 0:
            winning_ratio = test(game, agents)
            if winning_ratio > best_winning_ratio:
                best_winning_ratio = winning_ratio
                agents[0].save_model(FLAGS.model_dir)



if __name__ == '__main__':
    tf.app.run()