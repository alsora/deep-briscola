import tensorflow as tf

from agents.ai_agent import AIAgent
from agents.q_agent import QAgent
from agents.human_agent import HumanAgent

import environment as brisc
from utils import BriscolaLogger

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

    brisc.play_episode(game, agents)



if __name__ == '__main__':

    # Parameters
    # ==================================================

    # Model directory
    tf.flags.DEFINE_string("model_dir", "", "Provide a trained model path if you want to play against a deep agent (default: None)")

    FLAGS = tf.flags.FLAGS

    tf.app.run()

