import numpy as np
import tensorflow as tf
import itertools, time, random, os, shutil


class ReplayMemory:

    def __init__(self, capacity, features):

        n_features = features
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

    def __init__(self, n_actions, n_features):

        # init vars
        self.observed_state = {}
        self.learn_step_counter = 0
        self.wrong_move = False
        self.session = None
        self.state = None
        self.state_ = None
        self.action = None
        self.reward = None

        # network parameters
        # i.e. state is (None, n_features)
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = 1e-4
        self.batch_size = 100
        self.replace_target_iter = 2500

        # create network
        self.replay_memory = ReplayMemory(10000, self.n_features)
        self.create_network()
        self.initialize_session()


    def create_network(self):

        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='states_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e3')
            e4 = tf.layers.dense(e3, 32, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e4')

            self.q_eval = tf.layers.dense(e4, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q1')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t2, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t3')
            t4 = tf.layers.dense(t3, 32, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t4')

            self.q_next = tf.layers.dense(t4, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q2')

        with tf.variable_scope('predictions'):
            self.predictions = tf.argmax(self.q_eval, 1, output_type=tf.int32, name="argmax")
        # q_target is the discounted reward
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        # loss computed as difference between predicted q[a] and q_target
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


    def update(self, reward):

        '''
        if self.wrong_move:
            self.reward = -10
        else:
            self.reward = reward
        '''
        self.reward = reward

        if self.last_state is None:
            return

        self.replay_memory.push(self.last_state, self.action, self.reward, self.state)

        if self.replay_memory.size() < self.batch_size:
            return

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')

        batch_memory = self.replay_memory.sample(self.batch_size)

        _, loss = self.session.run(
            [self._train_op, self.loss,],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        if self.learn_step_counter % self.replace_target_iter == 0:
            print("Loss: ", loss)

        self.learn_step_counter += 1


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


    def save_model(self, output_dir = ''):

        if not output_dir:
            output_dir = 'saved_model'

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
        builder.add_meta_graph_and_variables(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            clear_devices=True)

        builder.save()


    def load_model(self, saved_model_dir = ''):

        if not saved_model_dir:
            saved_model_dir = 'saved_model'

        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

