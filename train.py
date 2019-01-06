import os
import argparse
import tensorflow as tf
import environment as brisc

from agents.random_agent import RandomAgent
from agents.q_agent import QAgent
from agents.ai_agent import AIAgent
from evaluate import evaluate
import environment as brisc
from utils import BriscolaLogger



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
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []
    agent = QAgent(
        FLAGS.epsilon,
        FLAGS.epsilon_increment,
        FLAGS.epsilon_max,
        FLAGS.discount,
        FLAGS.network,
        FLAGS.layers,
        FLAGS.learning_rate,
        FLAGS.replace_target_iter,
        FLAGS.batch_size)
    agents.append(agent)
    agent = RandomAgent()
    agents.append(agent)

    train(game, agents, FLAGS.num_epochs, FLAGS.evaluate_every, FLAGS.num_evaluations, FLAGS.model_dir)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--model_dir", default="saved_model", help="Where to save the trained model, checkpoints and stats", type=str)
    parser.add_argument("--num_epochs", default=100000, help="Number of training games played", type=int)

    # Reinforcement Learning parameters
    parser.add_argument("--epsilon", default=0, help="How likely is the agent to choose the best reward action over a random one", type=float)
    parser.add_argument("--epsilon_increment", default=5e-5, help="How much epsilon is increased after each action taken up to epsilon_max", type=float)
    parser.add_argument("--epsilon_max", default=0.85, help="The maximum value for the incremented epsilon", type=float)
    parser.add_argument("--discount", default=0.85, help="How much a reward is discounted after each step", type=float)

    # State parameters
    parser.add_argument("--cards_order", default="append", choices=['value', 'replace_last_used', 'append'], help="Where a drawn card is put in the hand")
    parser.add_argument("--cards_representation", default="hot_on_num_seed", choices=['hot_on_deck', 'hot_on_num_seed'], help="How to one-hot encode cards")
    parser.add_argument("--history", help="Include all seen cards in the state", action="store_true")

    # Network parameters
    parser.add_argument("--network", default="drqn", choices=['dqn', 'drqn'], help="Neural Network used for approximating value function")
    parser.add_argument('--layers', default=[256, 128], help="Definition of layers for the chosen network", type=int, nargs='+')
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for the network updates", type=float)
    parser.add_argument("--replace_target_iter", default=2000, help="Number of update steps before copying evaluation weights into target network", type=int)
    parser.add_argument("--batch_size", default=100, help="Training batch size", type=int)

    # Evaluation parameters
    parser.add_argument("--evaluate_every", default=1000, help="Evaluate model after this many epochs", type=int)
    parser.add_argument("--num_evaluations", default=500, help="Evaluate on these many episodes for each test", type=int)

    FLAGS = parser.parse_args()

    tf.app.run()
