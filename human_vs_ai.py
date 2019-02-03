import tensorflow as tf
import argparse

from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from agents.human_agent import HumanAgent

import environment as brisc
from utils import BriscolaLogger
from utils import NetworkTypes

def main(argv=None):

    # Initializing the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
    game = brisc.BriscolaGame(2, logger)

    # Initialize agents
    agents = []
    agents.append(HumanAgent())

    if FLAGS.model_dir:
        agent = QAgent()
        agent.load_model(FLAGS.model_dir)
        agent.make_greedy()
        agents.append(agent)
    else:
        agent = AIAgent()
        agents.append(agent)

    brisc.play_episode(game, agents, train=False)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, help="Provide a trained model path if you want to play against a deep agent", type=str)
    parser.add_argument("--network", default=NetworkTypes.DRQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN], help="Neural Network used for approximating value function")

    FLAGS = parser.parse_args()

    tf.app.run()


