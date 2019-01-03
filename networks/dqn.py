import numpy as np
import tensorflow as tf
import random, os, shutil


class ReplayMemory:

    def __init__(self, capacity, n_features):

        self.capacity = capacity
        # initialize zero memory, each sample in memory has size [s + a + r + s_]
        self.memory = np.zeros((self.capacity, n_features * 2 + 2))
        self.memory_counter = 0

    def push(self, s, a, r, s_):
        # stacks together all states element
        event = np.hstack((s, [a, r], s_))
        # get the index where to insert the event
        index = self.memory_counter % self.capacity
        self.memory[index, :] = event

        # increment memory_counter avoiding overflow
        self.memory_counter += 1
        if self.memory_counter == (self.capacity * 2):
            self.memory_counter = self.capacity

    def sample(self, batch_size):

        if self.memory_counter > self.capacity:
            # the replay memory is all written, so I can sample on all the array size [0, self.capacity]
            sample_index = np.random.choice(self.capacity, size=batch_size)
        else:
            # only part of the replay memory is written, so I sample up to where it's written [0, self.memory_counter]
            sample_index = np.random.choice(self.memory_counter, size=batch_size)

        batch_memory = self.memory[sample_index, :]
        return batch_memory

    def size(self):
        return min(self.capacity, self.memory_counter)


class DQN:

    def __init__(self, n_actions, n_features, learning_rate = 1e-3, discount = 0.85):

        # network parameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = discount
        self.batch_size = 100
        self.replace_target_iter = 2000

        # init vars
        self.learn_step_counter = 0
        self.wrong_move = False
        self.session = None

        # create replay memroy
        capacity = 10000
        self.replay_memory = ReplayMemory(capacity, self.n_features)

        # create network
        self.create_network()
        self.initialize_session()


    def create_network(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            # input placeholders
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='states_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # evaluation network
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e1')
                e2 = tf.layers.dense(e1, 64, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e2')
                e3 = tf.layers.dense(e2, 64, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e3')
                e4 = tf.layers.dense(e3, 32, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e4')

                self.q = tf.layers.dense(e4, self.n_actions, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='q')

            # target network
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
                                                bias_initializer=b_initializer, name='q_next')

            with tf.variable_scope('predictions'):
                # predicted actions according to evaluation network
                self.argmax_action = tf.argmax(self.q, 1, output_type=tf.int32, name="argmax")
            with tf.variable_scope('q_target'):
                # discounted reward on the target network
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='q_target')
                # stop gradient to avoid updating target network
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_wrt_a'):
                # q value of chosen action
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)
            with tf.variable_scope('loss'):
                # loss computed as difference between predicted q[a] and (current_reward + discount * q_target[best_future_action])
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_wrt_a, name='td_error'))
            with tf.variable_scope('train'):
                opt = tf.train.AdamOptimizer(self.learning_rate)
                grads_and_vars = opt.compute_gradients(self.loss)
                self._train_op = opt.apply_gradients(grads_and_vars, name="optimizer")

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            with tf.variable_scope('hard_replacement'):
                # operator for assiging evaluation network weights to the target network
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


    def get_q_table(self, state):
            ''' Compute q table for current state'''

            states_op = self.session.graph.get_operation_by_name("states").outputs[0]
            #argmax_op = self.session.graph.get_operation_by_name("predictions/argmax").outputs[0]
            q_op = self.session.graph.get_operation_by_name("eval_net/q/BiasAdd").outputs[0]

            input_state = np.expand_dims(state, axis=0)
            q = self.session.run([q_op], feed_dict={states_op: input_state})

            return q[0][0]


    def learn(self, last_state, action, reward, state):

        # I have a full event [s, a, r, s_], push it into replay memory
        self.replay_memory.push(last_state, action, reward, state)

        if self.replay_memory.size() < self.batch_size:
            # there are not enough samples for a training step in the replay memory
            return
        # get a batch of samples from replay memory
        batch_memory = self.replay_memory.sample(self.batch_size)

        # run a newtork training step
        _, loss = self.session.run(
            [self._train_op, self.loss,],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        # check if it's time to copy the target network into the evaluation network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            print("Loss: ", loss)

        self.learn_step_counter += 1


    def initialize_session(self):
        '''Defines self.sess and initialize the variables'''
        session_conf = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False)

        self.session = tf.Session(config = session_conf, graph=self.graph)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.session.run(self.init)


    def save_model(self, output_dir):
        '''Save the network graph and weights to disk'''
        if not output_dir:
            raise ValueError('You have to specify a valid output directory for DeepAgent.save_model')

        if not os.path.exists(output_dir):
            # if provided output_dir does not already exists, create it
            os.mkdir(output_dir)

        self.saver.save(self.session, "./" + output_dir + '/')


    def load_model(self, saved_model_dir):
        '''Initialize a new tensorflow session loading network and weights from a saved model'''
        self.saver.restore(self.session, "./" + saved_model_dir + '/')

