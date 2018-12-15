import tensorflow as tf

from agents_base.random_agent import RandomAgent
from agents.deep_agent import DeepAgent
from agents.ai_agent import AIAgent
import environment as brisc


# Parameters
# ==================================================

# Model directory
tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs")

# Deep Agent parameters
tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
tf.flags.DEFINE_float("epsilon_increment", 5e-6, "How much epsilon is increased after each action takenm up to 1 (default: 5e-6)")
tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremental epsilon (default: 0.85)")
tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

# Network parameters
tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


# Evaluation parameters
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate on these many episodes for each test (default: 500)")

FLAGS = tf.flags.FLAGS


def main(argv=None):

    # Initializing the environment
    game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)

    # Initialize agents
    agents = []
    agent = DeepAgent(FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount)
    agents.append(agent)
    agent = RandomAgent()
    agents.append(agent)

    best_winning_ratio = -1
    for epoch in range(1, FLAGS.num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')

        game_winner_id, winner_points = play_episode(game, agents)

        if epoch % FLAGS.evaluate_every == 0:
            victory_rates, average_points = evaluate(game, agents, FLAGS.num_evaluations)
            print("DeepAgent wins ", victory_rates[0], "% with average points ", average_points[0])
            if victory_rates[0] > best_winning_ratio:
                best_winning_ratio = victory_rates[0]
                agents[0].save_model(FLAGS.model_dir)



def play_episode(game, agents):

    game.reset()
    keep_playing = True
    while keep_playing:

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
        # update agents
        for i, player_id in enumerate(players_order):
            player = game.players[player_id]
            agent = agents[player_id]
            # agent observes new state after acting
            agent.observe(game, player, game.deck)

            reward = rewards[i]
            agent.update(reward)

        # update the environment
        keep_playing = game.draw_step()

    return game.end_game()


def evaluate(game, agents, num_evaluations):

    total_wins = [0] * len(agents)
    total_points = [0] * len(agents)
    victory_rates = [0] * len(agents)
    average_points = [0] * len(agents)

    for _ in range(num_evaluations):

        game.reset()
        keep_playing = True

        while keep_playing:

            players_order = game.get_players_order()
            for player_id in players_order:

                player = game.players[player_id]
                agent = agents[player_id]

                agent.observe(game, player, game.deck)
                available_actions = game.get_player_actions(player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)

            winner_player_id, points = game.evaluate_step()

            keep_playing = game.draw_step()

        game_winner_id, winner_points = game.end_game()

        total_wins[game_winner_id] += 1
        for player in game.players:
            total_points[player.id] += player.points

    for player in game.players:
        victory_rates[player.id] = (total_wins[player.id]/float(num_evaluations))*100
        average_points[player.id] = float(total_points[player.id])/float(num_evaluations)

    return victory_rates, average_points




if __name__ == '__main__':
    tf.app.run()