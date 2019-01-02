import tensorflow as tf
import tensorflow_probability as tfp
import gym
from collections import deque
from random import sample
import numpy as np
from datetime import datetime


class Policy:
    def __init__(self, sess, state_size, action_size, lr, batch_size, alpha_entropy):
        self.action_size = action_size
        self.state_size = state_size
        self.sess = sess

        with tf.variable_scope("Policy"):
            with tf.variable_scope("model"):
                self.state_input, model_output = self._create_model()
                self.mean_action = tf.math.tanh(model_output[..., action_size:])*2
                log_sigma = model_output[..., :action_size]

                dist = tfp.distributions.MultivariateNormalDiag(loc=self.mean_action, scale_diag=tf.exp(log_sigma))

                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Policy/model")
                self.action = dist.sample()

            with tf.variable_scope("probs"):
                self.probs = dist.prob(self.action)
                self.log_prob_op = tf.reshape(tf.log(self.probs),[-1, 1])

            with tf.variable_scope("training"):
                self.q_grad_a = tf.placeholder(tf.float32, [None, action_size])
                self.training_action = tf.placeholder(tf.float32, [None, action_size])

                self.training_probs = dist.prob(self.training_action)
                self.training_log_prob_op = tf.reshape(tf.log(self.training_probs),[-1, 1])

                self.log_prob_grad_a = tf.gradients(self.training_log_prob_op, self.training_action)

                self.entropy_summary = tf.summary.scalar("entropy", - tf.reduce_mean(self.training_log_prob_op*self.training_probs))

                #maximizes reward
                grad_s1 = tf.gradients(self.action, self.variables, (self.log_prob_grad_a-self.q_grad_a), name="grad_s1")
                #maximizes entropy
                grad_s2 = tf.gradients(self.training_log_prob_op, self.variables, name="grad_s2")

                with tf.variable_scope("gradient"):
                    self.grad = [(grad_s1[i] + alpha_entropy * grad_s2[i])/batch_size for i in range(len(grad_s1))]
                    self.clipped_grad,_ = tf.clip_by_global_norm(self.grad, 7)

                    self.c_grad_summaries = [tf.summary.histogram("pol_grads_{}".format(i), self.clipped_grad[i]) for i in range(len(self.clipped_grad))]

                    grads = zip(self.clipped_grad, self.variables)

                optimizer = tf.train.AdamOptimizer(lr)
                self.optimize = optimizer.apply_gradients(grads)

            self.summary_op = tf.summary.merge_all(scope="Policy/model")

    def log_probs_for_action(self, states, actions):
        return self.sess.run(self.log_prob_op, feed_dict={
            self.state_input: states,
            self.action_ph: actions
        })

    def get_summaries(self):
        return self.sess.run(self.summary_op)

    def predict(self, state):
        return sess.run([self.action, self.log_prob_op], feed_dict={
            self.state_input: state
        })

    def train(self, states, actions, q_grad_a):
        es, gs, _ =self.sess.run([self.entropy_summary, self.c_grad_summaries, self.optimize], feed_dict={
            self.q_grad_a:q_grad_a,
            self.state_input: states,
            self.training_action: actions
        })
        return es, gs

    def _create_model(self):
        layer_names = ["layer-1", "layer-2", "layer-3"]
        initializer = tf.initializers.random_uniform(-0.05, 0.05)

        state_input = tf.placeholder(tf.float32, [None, self.state_size])
        d1 = tf.layers.Dense(64, activation="relu", name=layer_names[0], kernel_initializer = initializer)(state_input)
        d2 = tf.layers.Dense(32, activation="relu", name=layer_names[1], kernel_initializer = initializer)(d1)
        output = tf.layers.Dense(2*self.action_size, name=layer_names[2], kernel_initializer = initializer)(d2)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return state_input, output


class StateValueApproximator:
    def __init__(self, sess, state_size, lr, tau, alpha):
        self.sess = sess
        self.state_size = state_size

        with tf.variable_scope("V_s"):
            with tf.variable_scope("model"):
                self.state_input, self.value_output = self._create_model()

            self.model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/model")

            with tf.variable_scope("target_model"):
                self.target_state_input, self.target_value_output = self._create_model()

            self.target_model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="V_s/target_model")

            with tf.variable_scope("target_copy"):
                self.copy_ops = [self.target_model_variables[i].assign(self.target_model_variables[i] * (1. - tau) + self.model_variables[i] *  tau)
                                 for i in range(len(self.model_variables))]

            with tf.variable_scope("loss"):
                self.q_value_ph = tf.placeholder(tf.float32, [None, 1])
                self.log_prob_ph = tf.placeholder(tf.float32, [None, 1])

                self.loss = tf.reduce_mean(tf.square(self.value_output - self.q_value_ph + alpha * self.log_prob_ph))
                self.loss_summary = tf.summary.scalar("loss", self.loss)

            with tf.variable_scope("training"):
                self.optimizer = tf.train.AdamOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.model_variables)
                self.clipped_grads,_ = tf.clip_by_global_norm(self.grads, 7)

                grad_var_pairs = zip(self.clipped_grads, self.model_variables)

                self.optimize = self.optimizer.apply_gradients(grad_var_pairs)

            self.summaries = tf.summary.merge_all(scope="V_s/model")

    def _create_model(self):
        layer_names = ["layer-1", "layer-2", "layer-3"]

        state_input = tf.placeholder(tf.float32, [None, self.state_size])
        d1 = tf.layers.Dense(64, activation="relu", name=layer_names[0])(state_input)
        d2 = tf.layers.Dense(32, activation="relu", name=layer_names[1])(d1)
        output = tf.layers.Dense(1, name=layer_names[2])(d2)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return state_input, output

    def get_summaries(self):
        return self.sess.run(self.summaries)

    def update_target(self):
        self.sess.run(self.copy_ops)

    def predict(self, states):
        return self.sess.run(self.value_output, feed_dict={
            self.state_input: states
        })

    def predict_target(self, states):
        return self.sess.run(self.target_value_output, feed_dict={
            self.target_state_input: states
        })

    def train(self, states, q_values, log_probs):
        loss_summary, _ = self.sess.run([self.loss_summary, self.optimize], feed_dict={
            self.state_input: states,
            self.q_value_ph: q_values,
            self.log_prob_ph: log_probs
        })
        return loss_summary


class QValueApproximator:
    def __init__(self, sess, name, state_size, action_size, lr):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        with tf.variable_scope(name):
            with tf.variable_scope("model"):
                self.state_input, self.action_input, self.output = self._create_model()

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/model".format(name))

            with tf.variable_scope("loss"):
                self.q_target_ph = tf.placeholder(tf.float32, [None, 1])
                self.loss = tf.reduce_mean(tf.square(self.output - self.q_target_ph))
                self.loss_summary = tf.summary.scalar("loss", self.loss)

            with tf.variable_scope("training"):
                optimizer = tf.train.AdamOptimizer(lr)
                self.grads = tf.gradients(self.loss, self.variables)
                self.clipped_grads,_ = tf.clip_by_global_norm(self.grads, 7)

                grad_var_pairs = zip(self.clipped_grads, self.variables)

                self.optimize = optimizer.apply_gradients(grad_var_pairs)

            with tf.variable_scope("grad_a"):
                self.q_summary = tf.summary.scalar("q_value", tf.reduce_mean(self.output))
                self.grad_a_op = tf.gradients(self.output, self.action_input)[0]

            self.summary_op = tf.summary.merge_all(scope=name+"/model")

    def _create_model(self):
        layer_names = ["layer-s1", "layer-s2", "layer-a1", "layer-m1", "layer-m2"]

        state_input = tf.placeholder(tf.float32, [None, self.state_size])
        ds1 = tf.layers.Dense(64, activation="relu", name=layer_names[0])(state_input)
        merge_s = tf.layers.Dense(64, name=layer_names[1])(ds1)

        action_input = tf.placeholder(tf.float32, [None, self.action_size])
        merge_a = tf.layers.Dense(64, name=layer_names[2])(action_input)

        merged = tf.add(merge_a, merge_s)
        dm1 = tf.layers.Dense(64, activation="relu", name=layer_names[3])(merged)
        output = tf.layers.Dense(1, name=layer_names[4])(dm1)

        for name in layer_names:
            with tf.variable_scope(name, reuse=True):
                tf.summary.histogram("kernel", tf.get_variable("kernel"))
                tf.summary.histogram("bias", tf.get_variable("bias"))

        return state_input, action_input, output

    def predict(self, states, actions):
        return self.sess.run(self.output, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })

    def get_summaries(self):
        return self.sess.run(self.summary_op)

    def grad_a(self, states, actions):
        return self.sess.run([self.grad_a_op, self.q_summary], feed_dict={
            self.state_input: states,
            self.action_input: actions
        })

    def train(self, states, actions, q_targets):
        loss_summary, _ =self.sess.run([self.loss_summary, self.optimize], feed_dict={
            self.state_input: states,
            self.action_input: actions,
            self.q_target_ph: q_targets
        })
        return loss_summary


# HYPERPARAMETERS

BATCH_SIZE = 128
ACTION_SIZE = 1
ALPHA = 0.03
STATE_SIZE = 3
LR = 3E-4
TAU = 0.005
GAMMA = 0.99

tb_verbose = False

with tf.Session() as sess:
    env = gym.make("Pendulum-v0")

    vs = StateValueApproximator(sess, STATE_SIZE, LR, TAU, ALPHA)
    pol = Policy(sess, STATE_SIZE, ACTION_SIZE, LR, BATCH_SIZE, ALPHA)
    q1 = QValueApproximator(sess, "Q1", STATE_SIZE, ACTION_SIZE, LR)
    q2 = QValueApproximator(sess, "Q2", STATE_SIZE, ACTION_SIZE, LR)

    now =datetime.now()

    # SET PATH
    writer = tf.summary.FileWriter("/home/florus/tb/t:{}-a:{}-lr:{}#{}".format(TAU, ALPHA, LR, now.strftime("%H:%M:%S")), sess.graph)

    buffer = deque(maxlen=200000)

    summary_val = tf.placeholder(tf.float32, [])
    reward_summary = tf.summary.scalar("reward", summary_val)

    init = tf.global_variables_initializer()
    sess.run(init)

    i_episode = 0
    steps = 0

    while True:
        state = env.reset()
        done = False
        episode_reward = 0
        i_episode += 1

        while not done:
            steps += 1

            env.render()

            action_array, _ = pol.predict(np.reshape(state, [1, -1]))
            action = action_array[0]
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward

            buffer.append((state, action, reward, next_state, done))

            if len(buffer) < BATCH_SIZE:
                continue

            batch = sample(buffer, BATCH_SIZE)
            states = np.array([s[0] for s in batch])
            actions = np.array([s[1] for s in batch])
            rewards = np.reshape([s[2] for s in batch], [-1 ,1])
            next_states = np.array([s[3] for s in batch])
            dones = np.reshape([s[4] for s in batch], [-1, 1])

            var_summaries = []
            loss_summaries = []

            on_policy_actions, log_probs = pol.predict(states)

            qval1 = q1.predict(states, on_policy_actions)
            qval2 = q2.predict(states, on_policy_actions)

            qvalmin = np.minimum(qval1, qval2)

            v_loss_summary = vs.train(states, qvalmin, log_probs)
            loss_summaries.append(v_loss_summary)

            v_targets = vs.predict_target(next_states)
            q_targets = rewards + GAMMA * v_targets * np.logical_not(dones)

            q1loss = q1.train(states, actions, q_targets)
            q2loss = q2.train(states, actions, q_targets)
            loss_summaries.extend([q1loss, q2loss])

            grad_as, q_summary = q1.grad_a(states, on_policy_actions)
            loss_summaries.append(q_summary)
            entropy_summary, pol_grad_summaries = pol.train(states, on_policy_actions, grad_as)
            loss_summaries.append(entropy_summary)

            vs.update_target()

            for ls in loss_summaries:
                writer.add_summary(ls, steps)

            if tb_verbose:
                var_summaries.extend(pol_grad_summaries)
                var_summaries.extend([
                    pol.get_summaries(),
                    q1.get_summaries(),
                    q2.get_summaries(),
                    vs.get_summaries()
                ])

            for summary in var_summaries:
                writer.add_summary(summary, steps)

            state = next_state

        rew_sum = sess.run(reward_summary, feed_dict={
            summary_val: episode_reward
        })
        writer.add_summary(rew_sum, steps)





