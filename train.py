import tensorflow as tf

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
import environment as brisc


def train(game, agents, num_epochs, evaluate_every, num_evaluations, model_dir = ""):

    best_total_wins = -1
    for epoch in range(1, num_epochs + 1):
        print ("Epoch: ", epoch, end='\r')

        game_winner_id, winner_points = brisc.play_episode(game, agents)

        if epoch % evaluate_every == 0:
            for agent in agents:
                agent.make_greedy()
            total_wins, points_history = evaluate(game, agents, num_evaluations)
            for agent in agents:
                agent.restore_epsilon()
            if total_wins[0] > best_total_wins:
                best_total_wins = total_wins[0]
                agents[0].save_model(model_dir)

    return best_total_wins



def main(argv=None):

    # Initializing the environment
    game = brisc.BriscolaGame(2, verbosity=brisc.LoggerLevels.TRAIN)

    # Initialize agents
    agents = []
    agent = QAgent(
        FLAGS.epsilon, FLAGS.epsilon_increment, FLAGS.epsilon_max, FLAGS.discount,
        FLAGS.learning_rate)
    agents.append(agent)
    agent = RandomAgent()
    agents.append(agent)

    train(game, agents, FLAGS.num_epochs, FLAGS.evaluate_every, FLAGS.num_evaluations, FLAGS.model_dir)


if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "saved_model", "Where to save the trained model, checkpoints and stats (default: pwd/saved_model)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs")

    # Deep Agent parameters
    tf.flags.DEFINE_float("epsilon", 0, "How likely is the agent to choose the best reward action over a random one (default: 0)")
    tf.flags.DEFINE_float("epsilon_increment", 5e-5, "How much epsilon is increased after each action taken up to epsilon_max (default: 5e-6)")
    tf.flags.DEFINE_float("epsilon_max", 0.85, "The maximum value for the incremented epsilon (default: 0.85)")
    tf.flags.DEFINE_float("discount", 0.85, "How much a reward is discounted after each step (default: 0.85)")

    # Network parameters
    tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate for the network updates (default: 1e-4)")


    # Evaluation parameters
    tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model after this many steps (default: 1000)")
    tf.flags.DEFINE_integer("num_evaluations", 500, "Evaluate on these many episodes for each test (default: 500)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()
