import tensorflow as tf

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
import environment as tictac

# Parameters
# ==================================================

# Model directory
tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs")

# Deep Agent parameters
tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
tf.flags.DEFINE_float("epsilon_increment", 5e-4, "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremented epsilon (default: 0.85)")
tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

# Network parameters
tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


# Evaluation parameters
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate on these many episodes for each test (default: 500)")

FLAGS = tf.flags.FLAGS

def main(argv=None):

    # Initializing the environment
    game = tictac.TicTacToeGame(2, verbosity=tictac.LoggerLevels.TRAIN)

    # Initialize agents
    agents = []
    agent = QAgent(
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)
    agents.append(agent)
    agent = RandomAgent()
    agents.append(agent)

    train(game, agents, FLAGS.num_epochs, FLAGS.evaluate_every, FLAGS.num_evaluations, FLAGS.model_dir)


def train(game, agents, num_epochs, evaluate_every, num_evaluations, model_dir = ""):

    best_winning_ratio = -1
    for epoch in range(1, num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')

        game_winner_id, winner_points = play_episode(game, agents)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            victory_rates, count_draws = evaluate(game, agents, num_evaluations)
            for agent in agents:
                agent.restore_epsilon()
            print("DeepAgent wins ", "{:.2f}".format(victory_rates[0]), "% with count_draws  ", count_draws)
            print ("reward = ", (100* victory_rates[0] * num_evaluations/100.0 - victory_rates[1] * num_evaluations/100.0  + 10 * count_draws)/num_evaluations)
            if victory_rates[0] > best_winning_ratio:
                best_winning_ratio = victory_rates[0]
                agents[0].save_model(model_dir)

    return best_winning_ratio



def play_episode(game, agents):

    game.reset()
    keep_playing = True
    winner_player_id = -1
    while keep_playing:

        # action step
        players_order = game.get_players_order()
        rewards = [0, 0]
        for player_id in players_order:

            agent = agents[player_id]
            # agent observes state before acting
            agent.observe(game, player_id)
            available_actions = game.get_player_actions(game.board, player_id)
            action = agent.select_action(available_actions)

            game.play_step(action, player_id)

            win, draw = game.evaluate_step(player_id)

            if win:
                winner_player_id = player_id
                rewards[player_id] = 100
                rewards[1 - player_id] = -1
                keep_playing = False
                break
            elif draw:
                rewards[0] = rewards[1] = 0
                keep_playing = False
                break

        # update agents
        for player_id in players_order:
            agent = agents[player_id]
            # agent observes new state after acting
            agent.observe(game, player_id)
            reward = rewards[player_id]
            agent.update(reward)

    return winner_player_id, 0


def evaluate(game, agents, num_evaluations):

    total_wins = [0] * len(agents)
    victory_rates = [0] * len(agents)

    count_draws = 0

    for _ in range(num_evaluations):

        game.reset()
        keep_playing = True

        while keep_playing:

            players_order = game.get_players_order()
            for player_id in players_order:

                agent = agents[player_id]
                # agent observes state before acting
                agent.observe(game, player_id)
                available_actions = game.get_player_actions(game.board, player_id)
                action = agent.select_action(available_actions)

                game.play_step(action, player_id)

                win, draw = game.evaluate_step(player_id)

                if win:
                  total_wins[player_id] += 1
                  keep_playing = False
                  break
                elif draw:
                  keep_playing = False
                  count_draws += 1
                  break

    print (total_wins)
    for player_id in range(2):
        victory_rates[player_id] = (total_wins[player_id]/float(num_evaluations))*100

    return victory_rates, count_draws




if __name__ == '__main__':
    tf.app.run()