import tensorflow as tf
import tensorflow_probability as tfp
from datetime import datetime
import gym
import numpy as np
import time

GRADIENT_NORM = 5


class Policy:
    def __init__(self, sess, state_size, action_size, lr, alpha_entropy,  epsilon):
        self.sess = sess
        self.action_size = action_size

        with tf.variable_scope("Sample_Policy"), tf.device("/cpu:0"):
            self.sample_state_ph = tf.placeholder(tf.float32, [None, state_size], name="sample_state_ph")
            self.sample_pi, self.mean_action = self._create_model(trainable=False, input=self.sample_state_ph)
            self.sample_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Sample_Policy")
            self.sample_op = self.sample_pi.sample()

        with tf.variable_scope("Policy"):
            self.state_ph = tf.placeholder(tf.float32, [None, state_size], name="state_ph")
            self.action_ph = tf.placeholder(tf.float32, [None, action_size], name="action_ph")
            self.advantage_ph = tf.placeholder(tf.float32, [None, 1], name="action_ph")

            with tf.variable_scope("pi"):
                self.pi, _ = self._create_model(trainable=True, input=self.state_ph)
                self.pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/pi")

            with tf.variable_scope("old_pi"):
                self.old_pi, _ = self._create_model(trainable=False, input=self.state_ph)
                self.old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/old_pi")

            with tf.variable_scope("loss"):
                prob_ratio = self.pi.prob(self.action_ph) / self.old_pi.prob(self.action_ph)
                surrogate = prob_ratio * self.advantage_ph

                clipped_surrogate = tf.minimum(surrogate,  tf.clip_by_value(prob_ratio, 1.-epsilon, 1.+epsilon)*self.advantage_ph)

                self.pi_entropy = self.pi.entropy()

                tf.summary.scalar("entropy", tf.reduce_mean(self.pi_entropy))

                self.loss = -tf.reduce_mean(clipped_surrogate + alpha_entropy * self.pi_entropy)

                tf.summary.scalar("objective", self.loss)

            with tf.variable_scope("training"):
                self.gradients = tf.gradients(self.loss, self.pi_vars)
                #self.gradients = [tf.clip_by_value(g, -1000, 1000) for g in self.gradients]
                #self.gradients, _ = tf.clip_by_global_norm(self.gradients, GRADIENT_NORM)
                grads = zip(self.gradients, self.pi_vars)
                self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

                [tf.summary.histogram(v.name, g) for g, v in grads]

            with tf.variable_scope("update_weights"):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_vars, self.old_pi_vars)]
                self.update_sample_pi_op = [samp.assign(p) for p, samp in zip(self.pi_vars, self.sample_pi_vars)]

            self.summary_op = tf.summary.merge_all(scope="Policy")

    def _create_model(self, trainable, input):
        layer_names = ["l1", "l2", "l3", "l4"]

        l1 = tf.layers.Dense(32, activation="relu", name=layer_names[0], trainable=trainable, kernel_initializer = tf.initializers.he_normal(),)(input)
        l2 = tf.layers.Dense(64, activation="relu", name=layer_names[1], trainable=trainable, kernel_initializer = tf.initializers.he_normal(),)(l1)
        l3 = tf.layers.Dense(32, activation="relu", name=layer_names[2], trainable=trainable, activity_regularizer= tf.contrib.layers.l2_regularizer(scale=0.001),kernel_initializer = tf.initializers.he_normal(),)(l2)
        mu = tf.layers.Dense(self.action_size, activation="tanh", name=layer_names[3], trainable=trainable, kernel_initializer = tf.initializers.he_normal(),)(l3)

        log_sigma = tf.Variable(initial_value=tf.fill((self.action_size,), 0.), trainable=trainable)

        distribution = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))

        tf.summary.histogram("log_sigma", log_sigma)
        tf.summary.histogram("mu", mu)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return distribution, mu

    def sample_action(self, state):
        return self.sess.run([self.mean_action, self.sample_op], feed_dict={
            self.sample_state_ph: state
        })

    def train(self, states, actions, advantages):
        _, summaries =self.sess.run([self.optimize, self.summary_op], feed_dict={
            self.state_ph:states,
            self.action_ph: actions,
            self.advantage_ph: advantages
        })
        return summaries

    def update_network_weights(self):
        self.sess.run([self.update_oldpi_op, self.update_sample_pi_op])


class StateValueApproximator:
    def __init__(self, sess, state_size, lr, kernel_reg):
        self.sess = sess
        self.kernel_reg = kernel_reg

        with tf.variable_scope("V_s"):
            self.v_target_ph = tf.placeholder(tf.float32, [None, 1])
            self.state_ph = tf.placeholder(tf.float32, [None, state_size])

            with tf.variable_scope("model"):
                self.value_output = self._create_model()

                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/model")

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.square(self.value_output - self.v_target_ph))
                self.loss_summary = tf.summary.scalar("loss", self.loss)

            with tf.variable_scope("training"):
                self.optimizer = tf.train.AdamOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.variables)
                # self.grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
                #self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, GRADIENT_NORM)

                grad_var_pairs = zip(self.grads, self.variables)

                self.optimize = self.optimizer.apply_gradients(grad_var_pairs)

        self.summaries = tf.summary.merge_all(scope="V_s/model")

    def _create_model(self):
        layer_names = ["layer-1", "layer-2", "layer-3"]

        d1 = tf.layers.Dense(64, activation="relu", name=layer_names[0], kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(self.state_ph)
        d2 = tf.layers.Dense(64, activation="relu", name=layer_names[1], kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(d1)
        output = tf.layers.Dense(1, name=layer_names[2], kernel_initializer = tf.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.kernel_reg))(d2)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return output

    def get_summaries(self):
        return self.sess.run(self.summaries)

    def predict(self, states):
        return self.sess.run(self.value_output, feed_dict={
            self.state_ph: states
        })

    def train(self, states, v_targets):
        loss_summary, _ = self.sess.run([self.loss_summary, self.optimize], feed_dict={
            self.state_ph: states,
            self.v_target_ph: v_targets
        })
        return loss_summary


# HYPERPARAMETERS
BATCH_SIZE = 1024
HORIZON = 50

ALPHA = 0.005
A_LR = 1e-4
C_LR = 2e-4

EVAL_EVERY = 10

# These hyperparameters can be left as they are
KERNEL_REG = 0.0
EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.95

ACTION_BOUND = 2
ACTION_SIZE = 1
STATE_SIZE = 3

K_EPOCHS = 10

tb_verbose = True

with tf.Session() as sess:
    env = gym.make("Pendulum-v0")

    vs = StateValueApproximator(sess, STATE_SIZE, C_LR, KERNEL_REG)
    pol = Policy(sess, STATE_SIZE, ACTION_SIZE, A_LR, ALPHA, EPSILON)

    now = datetime.now()

    # SET PATH
    writer = tf.summary.FileWriter("/home/florus/tb-p4/a:{}-lr:{}#{}".format(ALPHA, A_LR, now.strftime("%H:%M:%S")), sess.graph)

    buffer = []

    batch_buffer = []
    batch_buffer_gae = []

    summary_val = tf.placeholder(tf.float32, [])
    reward_summary = tf.summary.scalar("reward", summary_val)
    eval_reward_summary = tf.summary.scalar("eval_reward", summary_val)

    init = tf.global_variables_initializer()
    sess.run(init)
    pol.update_network_weights()

    i_episode = 0
    steps = 0
    i_epochs = 0

    t = time.time()
    for _ in range(250):
        state = env.reset()
        done = False
        episode_reward = 0
        i_episode += 1

        is_eval_episode = i_episode % EVAL_EVERY == 0

        while not done:

            steps += 1
            #if is_eval_episode:
             #   env.render()

            mean_action, noisy_action = pol.sample_action(np.reshape(state, [1, -1]))

            # post processing separate from Model actions
            scaled_action = (mean_action if is_eval_episode else noisy_action)[0] * ACTION_BOUND
            next_state, reward, done, _ = env.step(scaled_action)

            episode_reward += reward

            if not is_eval_episode:
                buffer.append((state, noisy_action[0], reward, next_state, done))

            state = next_state

            if len(buffer) == HORIZON or (done and buffer):

                ## TRAINING
                states = np.array([s[0] for s in buffer])
                actions = np.array([s[1] for s in buffer])
                rewards = np.reshape([s[2] for s in buffer], [-1 ,1])
                next_states = np.array([s[3] for s in buffer])

                state_values = vs.predict(states)
                next_state_values = vs.predict(next_states)

                td_residuals = rewards + GAMMA * next_state_values - state_values

                gae_values = []
                last_gea = 0

                for tdr in reversed(td_residuals):
                    gae = tdr + LAMBDA * GAMMA * last_gea
                    gae_values.append(gae)
                    last_gea = gae

                gae_values.reverse()

                batch_buffer_gae.extend(gae_values)
                batch_buffer.extend(buffer)

                buffer = []

            if len(batch_buffer) >= BATCH_SIZE:
                states = np.array([s[0] for s in batch_buffer])
                actions = np.array([s[1] for s in batch_buffer])
                rewards = np.reshape([s[2] for s in batch_buffer], [-1 ,1])
                next_states = np.array([s[3] for s in batch_buffer])

                next_state_values = vs.predict(next_states)

                v_targets = rewards + GAMMA * next_state_values

                for _ in range(K_EPOCHS):
                    i_epochs += 1
                    pol_summaries = pol.train(states, actions, batch_buffer_gae)
                    v_summaries = vs.train(states, v_targets)
                    writer.add_summary(pol_summaries, i_epochs)
                    writer.add_summary(v_summaries, i_epochs)

                print("Batch size: {} ".format(len(batch_buffer)))

                pol.update_network_weights()
                batch_buffer = []
                batch_buffer_gae = []

        rew_sum = sess.run((eval_reward_summary if is_eval_episode else reward_summary), feed_dict={
            summary_val: episode_reward
        })
        writer.add_summary(rew_sum, steps)

    print(time.time() - t)






