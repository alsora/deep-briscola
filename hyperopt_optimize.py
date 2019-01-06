from hyperopt import hp, tpe, fmin, space_eval
import tensorflow as tf

from agents.q_agent import QAgent
from agents.random_agent import RandomAgent
import environment as brisc
from train import train
from utils import BriscolaLogger


space = {
    'discount': hp.choice('discount', [0.25, 0.75, 0.85, 0.9, 0.95]),
    'epsilon': hp.choice('epsilon', [0, 0.75]),
    'epsilon_increment' : hp.choice('epsilon_increment', [1e-8, 1e-7, 5e-6, 1e-6, 1e-5]),
    'epsilon_max' : hp.choice('epsilon_max', [0.8, 0.85, 0.9, 0.95]),
    'learning_rate' : hp.choice('learning_rate', [1e-5, 1e-4, 1e-3])
}

NUM_EPOCHS=50 * 1000
EVALUATE_EVERY=10 * 1000
EVALUATE_FOR=1000
MODEL_DIR='hyperopt_best_model'

def train_agent(hype_space):

    print("----------------------")
    print("Evaluating model: ", hype_space)

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = brisc.BriscolaGame(2, logger)

    tf.reset_default_graph()

    # Initialize agents
    agents = []
    agent = QAgent(
        hype_space['epsilon'], hype_space['epsilon_increment'], hype_space['epsilon_max'], hype_space['discount'],
        hype_space['learning_rate'])

    agents.append(agent)
    agents.append(RandomAgent())

    best_total_wins = train(game, agents, NUM_EPOCHS, EVALUATE_EVERY, EVALUATE_FOR, MODEL_DIR)

    print ("Best total wins ----->", best_total_wins)
    best_total_loses = EVALUATE_FOR - best_total_loses
    return best_total_loses



if __name__ == "__main__":

    # returns list of indices of parameter choices
    best_model = fmin(
        train_agent,
        space,
        algo=tpe.suggest,
        max_evals=100
    )

    print(best_model)
    print ("Best model is:")
    print(space_eval(space, best_model))
