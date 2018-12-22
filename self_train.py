import tensorflow as tf

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
import environment as brisc


victory_rates_hist  = []
average_points_hist = []



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
    points_history = [ [] for i in range(len(agents))]

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

        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id:
                total_wins[player.id] += 1

    return total_wins, points_history



def train(game, agents, num_epochs, evaluate_every, num_evaluations, model_dir = ""):

    best_winning_ratio = -1
    for epoch in range(1, num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')

        game_winner_id, winner_points = play_episode(game, agents)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            victory_rates, average_points = evaluate(game, agents, num_evaluations)
            victory_rates_hist.append(victory_rates)
            average_points_hist.append(average_points)

            for agent in agents:
                agent.restore_epsilon()
            #print("DeepAgent wins ", "{:.2f}".format(victory_rates[0]), "% with average points ", "{:.2f}".format(average_points[0]))
            if victory_rates[0] > best_winning_ratio:
                best_winning_ratio = victory_rates[0]
                agents[0].save_model(model_dir)

    return best_winning_ratio




def main(argv=None):

    # Initializing the environment
    game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)

    # Initialize agents
    agents = []
    agent = QAgent( 1,
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)
    agents.append(agent)
    agent = QAgent( 2,
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)
    agents.append(agent)

    train(game, agents, FLAGS.num_epochs, FLAGS.evaluate_every, FLAGS.num_evaluations, FLAGS.model_dir)


if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 121, "Number of training epochs")

    # Deep Agent parameters
    tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
    tf.flags.DEFINE_float("epsilon_increment", 5e-5, "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
    tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremented epsilon (default: 0.85)")
    tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

    # Network parameters
    tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


    # Evaluation parameters
    tf.flags.DEFINE_integer("evaluate_every", 30, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate on these many episodes for each test (default: 500)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()
































































