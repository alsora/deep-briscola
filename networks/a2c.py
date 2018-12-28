import numpy as np
import tensorflow as tf
import random, os, shutil


class A2C:

    def __init__(self, n_actions, n_features, learning_rate = 1e-3, discount = 0.85):

        # network parameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = discount
        self.batch_size = 100
        self.replace_target_iter = 2000

        self.entropy_coeff = 0.01
        self.value_function_coeff = 0.5

        # init vars
        self.learn_step_counter = 0
        self.wrong_move = False
        self.session = None

        # create network
        self.create_network()
        self.initialize_session()


    def create_network(self):

        # input placeholders
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='states')  # input State
        self.a = tf.placeholder(tf.int32, [None], name='actions')
        self.r = tf.placeholder(tf.float32, [None], name='rewards')
        self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
        #self.learning_rate = tf.placeholder(tf.float32, []) #TODO: do I need a placeholder?
        #self.is_training = tf.placeholder(tf.bool) #TODO: integrate with make_greedy also for other agents/networks

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # policy evaluation
        with tf.variable_scope('policy'):
            e1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, 64, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e3')
            e4 = tf.layers.dense(e3, 32, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e4')

            self.policy_logits = tf.layers.dense(e4, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='policy_logits')

            self.value_function = tf.layers.dense(e4, 1, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='value')

        #TODO: move it somewhere else
        with tf.variable_scope('value'):
            self.value_s = self.value_function[:, 0]

        #TODO: this can be done by the agent using self.policy_logits
        with tf.variable_scope('action'):
            noise = tf.random_uniform(tf.shape(self.policy_logits))
            self.action_s = tf.argmax(self.policy_logits - tf.log(-tf.log(noise)), 1)

        with tf.variable_scope('loss'):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.policy_logits,
                labels=self.a)

            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
            # mean squared error as predicted - ground truth
            mse = tf.square(tf.squeeze(self.value_function) - self.r) / 2.
            self.value_function_loss = tf.reduce_mean(mse)

            # entropy computation from openAI baselines
            a0 = self.policy_logits - tf.reduce_max(self.policy_logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            self.entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), 1))

            self.loss = self.policy_gradient_loss - self.entropy * self.entropy_coeff + self.value_function_loss * self.value_function_coeff

        with tf.variable_scope('train'):
            opt = tf.train.RMSPropOptimizer(self.learning_rate) #TODO: add decay, epsilon params
            grads_and_vars = opt.compute_gradients(self.loss)
            self._train_op = opt.apply_gradients(grads_and_vars, name="optimizer")


    #TODO: rename to something general, as "get_actions_policy"
    def get_q_table(self, state):
            ''' Compute q table for current state'''

            #states_op = self.session.graph.get_operation_by_name("states").outputs[0]
            #action_op = self.session.graph.get_operation_by_name("eval_net/q/BiasAdd").outputs[0]

            input_state = np.expand_dims(state, axis=0)
            #TODO: use operands instead of self. to identify tf operations
            policy, value = self.session.run([self.policy_logits, self.value_function], feed_dict={self.s: input_state})

            return policy[0], value[0]


    def learn(self, last_state, action, reward, value):

        advantage = reward - value

        #TODO: this should use minibatch
        #TODO: reward should be discounted

        # this is temporary, removed when using minibatch
        input_s = np.expand_dims(last_state, axis=0)
        input_a = np.expand_dims(action, axis=0)
        input_r = np.expand_dims(reward, axis=0)

        # run a newtork training step
        loss, policy_loss, value_loss, policy_entropy, _ = self.session.run(
            [self.loss, self.policy_gradient_loss, self.value_function_loss, self.entropy, self._train_op],
            feed_dict={
                self.s: input_s,
                self.a: input_a,
                self.r: input_r,
                self.advantage: advantage
            })


        self.learn_step_counter += 1



    def initialize_session(self):
        '''Defines self.sess and initialize the variables'''
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


    def save_model(self, output_dir):
        '''Save the network graph and weights to disk'''
        if not output_dir:
            raise ValueError('You have to specify a valid output directory for DeepAgent.save_model')

        if os.path.exists(output_dir):
            # if provided output_dir already exists, remove it
            shutil.rmtree(output_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
        builder.add_meta_graph_and_variables(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            clear_devices=True)
        # create a new directory output_dir and store the saved model in it
        builder.save()


    def load_model(self, saved_model_dir):
        '''Initialize a new tensorflow session loading network and weights from a saved model'''
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

