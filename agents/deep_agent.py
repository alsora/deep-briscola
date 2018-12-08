import numpy as np
import tensorflow as tf

__ACTIONS__ = [0, 1, 2]

class DeepAgent:

    def __init__(self):
        self.observed_state = {}
        self.session = None
        self.create_network()
        self.initialize_session()
        self.epsilon = 0.6
        self.wrong_move = False

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        #self.logger.info("Initializing tf session")
        session_conf = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False)
        self.session = tf.Session(config = session_conf)
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
        except:
            pass

    def create_network(self):
        self.input_x = tf.placeholder(tf.float32, [1, 40], name="input_x")
        self.num_actions = len(__ACTIONS__)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = 1e-3

        W1 = tf.get_variable("W1", shape=[self.input_x.shape[1], 128], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[128], dtype=tf.float32, initializer=tf.zeros_initializer())
        res1 = tf.nn.xw_plus_b(self.input_x, W1, b1, name="layer_1")

        W2 = tf.get_variable("W2", shape=[res1.shape[1], 64], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[64], dtype=tf.float32, initializer=tf.zeros_initializer())
        res2 = tf.nn.xw_plus_b(res1, W2, b2, name="layer_2")

        W3 = tf.get_variable("W3", shape=[res2.shape[1], 32], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", shape=[32], dtype=tf.float32, initializer=tf.zeros_initializer())
        res3 = tf.nn.xw_plus_b(res2, W3, b3, name="layer_3")

        W = tf.get_variable("W", shape=[res3.shape[1], self.num_actions], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[self.num_actions], dtype=tf.float32, initializer=tf.zeros_initializer())
        q_table = tf.nn.xw_plus_b(res3, W, b, name="q_table")

        predictions = tf.argmax(q_table, 1, output_type=tf.int32, name="predictions")

        next_q = tf.placeholder(tf.float32, [1, self.num_actions], name="next_q")
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=next_q, labels=q_table, name="loss")

        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads_and_vars = opt.compute_gradients(loss)
        optimizer = opt.apply_gradients(grads_and_vars, global_step=self.global_step, name="optimizer")




    def observe_game_state(self, game, deck):
        self.observed_state['turn_state'] = game.get_turn_state()
        self.observed_state['briscola_seed'] = game.briscola_seed
        self.observed_state['briscola_one_hot'] = deck.get_card_one_hot(game.briscola.id)
        self.observed_state['played_cards_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['turn_state']['played_cards']])

    def observe_player_state(self, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        self.observed_state['hand_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['hand']])

    def select_action(self, actions):

        hand_one_hot = self.observed_state['hand_one_hot']
        briscola_one_hot = self.observed_state['briscola_one_hot']
        played_cards_one_hot = self.observed_state['played_cards_one_hot']

        # state is an array [1,40] with all zeros and: 1 for cards in hand, 2 for briscola, 3 for played cards
        state = np.array(hand_one_hot)
        state += 2 * np.array(briscola_one_hot)
        state += 3 * np.array(played_cards_one_hot)
        state = np.expand_dims(state, axis=0)

        input_x_op = self.session.graph.get_operation_by_name("input_x").outputs[0]
        global_step_op = self.session.graph.get_operation_by_name("global_step").outputs[0]
        q_table_op = self.session.graph.get_operation_by_name("q_table").outputs[0]
        predictions_op = self.session.graph.get_operation_by_name("predictions").outputs[0]

        _, self.q_table, predictions = self.session.run([global_step_op, q_table_op, predictions_op], feed_dict={input_x_op: state})

        if np.random.rand(1) > self.epsilon:
            action = np.random.choice(actions)
        else:
            action = predictions[0]

        if action >= len(actions):
            print ("Selected invalid action!!!")
            self.wrong_move = True
            action = np.random.choice(actions)

        print("NETWORK RUN!")
        print ("PREDICTIONS---->", predictions)
        print ("Selected action: ", action, " among available ", actions)

        self.last_state = state
        self.last_action = action

        return action


    def update(self, reward, available_actions):

        hand_one_hot = self.observed_state['hand_one_hot']
        briscola_one_hot = self.observed_state['briscola_one_hot']
        played_cards_one_hot = self.observed_state['played_cards_one_hot']

        # state is an array [1,40] with all zeros and: 1 for cards in hand, 2 for briscola, 3 for played cards
        state = np.array(hand_one_hot)
        state += 2 * np.array(briscola_one_hot)
        state += 3 * np.array(played_cards_one_hot)
        state = np.expand_dims(state, axis=0)

        input_x_op = self.session.graph.get_operation_by_name("input_x").outputs[0]
        global_step_op = self.session.graph.get_operation_by_name("global_step").outputs[0]
        q_table_op = self.session.graph.get_operation_by_name("q_table").outputs[0]

        _, q_table_1 = self.session.run([global_step_op, q_table_op], feed_dict={input_x_op: state})

        max_q_1 = np.max(q_table_1)
        target_q = self.q_table
        target_q[0, self.last_action] = reward + 0.99 * max_q_1

        next_q_op = self.session.graph.get_operation_by_name("next_q").outputs[0]
        optimizer_op = self.session.graph.get_operation_by_name("optimizer").outputs[0]

        self.session.run([global_step_op, optimizer_op], feed_dict={input_x_op: self.last_state, next_q_op: target_q})



