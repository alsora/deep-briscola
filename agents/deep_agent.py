import numpy as np
import tensorflow as tf
import itertools, time, random, os

__ACTIONS__ = [0, 1, 2]

class ReplayMemory:

    def __init__(self,capacity):

        n_features = 120
        # initialize zero memory [s, a, r, s_]
        self.capacity = capacity
        self.memory = np.zeros((self.capacity, n_features * 2 + 2))
        self.memory_counter = 0

    def push(self, s, a, r, s_):
        event = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.capacity
        self.memory[index, :] = event
        self.memory_counter += 1

    def sample(self, batch_size):

        # sample batch memory from all written memory
        if self.memory_counter > self.capacity:
            sample_index = np.random.choice(self.capacity, size=batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=batch_size)

        batch_memory = self.memory[sample_index, :]
        return batch_memory

    def size(self):
        return min(self.capacity, self.memory_counter)


class DeepAgent:

    def __init__(self):
        self.observed_state = {}
        self.epsilon = 0.9 # grediness
        self.gamma = 0.9 # reward decay
        self.wrong_move = False
        self.session = None
        self.learn_step_counter = 0
        self.learning_rate = 1e-3

        self.state = None
        self.state_ = None
        self.action = None
        self.reward = None

        self.replay_memory = ReplayMemory(10000)
        self.create_network()
        self.initialize_session()



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

        self.n_features = 120
        self.n_actions = 3

        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='states_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 512, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e3')

            self.q_eval = tf.layers.dense(e3, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q1')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 512, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t2, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t3')

            self.q_next = tf.layers.dense(t3, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q2')

        with tf.variable_scope('predictions'):
            self.predictions = tf.argmax(self.q_eval, 1, output_type=tf.int32, name="argmax")
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = opt.compute_gradients(self.loss)
            self._train_op = opt.apply_gradients(grads_and_vars, name="optimizer")

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]



    def observe(self, game, player, deck):
        self.observed_state['hand'] = player.get_player_state()
        self.observed_state['hand_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['hand']])
        self.observed_state['turn_state'] = game.get_turn_state()
        self.observed_state['briscola_seed'] = game.briscola_seed
        self.observed_state['briscola_one_hot'] = deck.get_card_one_hot(game.briscola.id)
        self.observed_state['played_cards_one_hot'] = deck.get_cards_one_hot([card.id for card in self.observed_state['turn_state']['played_cards']])

        hand_one_hot = np.array(self.observed_state['hand_one_hot'])
        briscola_one_hot = np.array(self.observed_state['briscola_one_hot'])
        played_cards_one_hot = np.array(self.observed_state['played_cards_one_hot'])

        state = np.concatenate((hand_one_hot, played_cards_one_hot), axis=0)
        state = np.concatenate((state, briscola_one_hot), axis=0)

        self.last_state = self.state
        self.state = state



    def get_state(self):

        hand_one_hot = np.array(self.observed_state['hand_one_hot'])
        briscola_one_hot = np.array(self.observed_state['briscola_one_hot'])
        played_cards_one_hot = np.array(self.observed_state['played_cards_one_hot'])

        state = np.concatenate((hand_one_hot, played_cards_one_hot), axis=0)
        state = np.concatenate((state, briscola_one_hot), axis=0)
        state = np.expand_dims(state, axis=0)

        return state


    def select_action(self, actions):

        if np.random.uniform() < self.epsilon:
            states_op = self.session.graph.get_operation_by_name("states").outputs[0]
            predictions_op = self.session.graph.get_operation_by_name("predictions/argmax").outputs[0]

            input_state = np.expand_dims(self.state, axis=0)
            predictions = self.session.run([predictions_op], feed_dict={states_op: input_state})

            action = predictions[0][0]
        else:
            action = np.random.choice(actions)

        if action >= len(actions):
            #print ("Selected invalid action!!!")
            self.wrong_move = True
            action = np.random.choice(actions)

        self.action = action

        return action


    def forward_pass(self, states):
        pass


    def update(self, reward, available_actions):

        batch_size = 100
        n_features = 120
        self.replace_target_iter = 300

        self.reward = reward

        self.replay_memory.push(self.last_state, self.action, self.reward, self.state)

        if self.replay_memory.size() < batch_size:
            return

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        batch_memory = self.replay_memory.sample(batch_size)

        _, cost = self.session.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, : n_features],
                self.a: batch_memory[:, n_features],
                self.r: batch_memory[:, n_features + 1],
                self.s_: batch_memory[:, -n_features:],
        })

        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1



