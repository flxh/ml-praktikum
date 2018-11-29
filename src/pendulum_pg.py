"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense,  Input, Lambda
from keras.layers.merge import Add
from keras.losses import MSE
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as K
import sys

import tensorflow as tf

import random
from collections import deque


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class REINFORCEAgent:

    def __init__(self, session, learning_rate, action_size, state_size, action_scalar=1.):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        self.action_scalar = action_scalar

        self.sess = session

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.95
        self.learning_rate = learning_rate

        # Model for policy network
        self.model_input, self.model = self.create_model()
        self.reward_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.loss = - tf.reduce_mean((self.action_holder - self.model.output) * self.reward_holder)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimize_op = optimizer.minimize(self.loss)

    def create_model(self):
        initializer = RandomUniform(-0.005, 0.005)

        state_input = Input(shape=[self.state_size])
        h1 = Dense(128, activation='relu', kernel_initializer=initializer)(state_input)
        h2 = Dense(128, activation='relu', kernel_initializer=initializer)(h1)
        h3 = Dense(64, activation='relu', kernel_initializer=initializer)(h2)
        output = Dense(self.action_size, activation='tanh', kernel_initializer=initializer)(h3)
        scaled_output = Lambda(lambda x: x * self.action_scalar)(output)

        model = Model(input=state_input, output=scaled_output)
        return state_input, model

    # Use the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        return self.model.predict(state)

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network every episode
    def train(self, samples):
        states = np.asarray([e[0] for e in samples])
        actions = np.asarray([e[1] for e in samples])
        rewards = np.asarray([e[2] for e in samples])

        discounted_rewards = self.discount_rewards(rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            print('std = 0!')
            return 0

        discounted_rewards = np.reshape(discounted_rewards, [len(discounted_rewards), 1])

        return self.sess.run(self.optimize_op, feed_dict={
            self.model_input: states,
            self.action_holder: actions,
            self.reward_holder: discounted_rewards
        })

    def load_weights(self, path):
        self.model.load_weights(path)

def main():
    BUFFER_SIZE = 20000
    EPSILON_DECAY = 0.9999
    EPSILON_MIN = 0.05
    ACTION_SCALAR = 2
    AGENT_LR = 0.001

    agent_path = "agent_pend_pg.h5"

    is_training = False

    try:
        is_training = sys.argv[1] == "train"
    except:
        pass

    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    K.set_session(sess)
    agent = REINFORCEAgent(sess, AGENT_LR, action_dim, state_dim, action_scalar = ACTION_SCALAR)

    print("TRAINING" if is_training else "RUN")

    if not is_training:
        agent.load_weights(agent_path)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    i_episode = 0
    epsilon = 1.

    while True:
        i_episode += 1

        print("Episode : " + str(i_episode) + " Replay Buffer " + str(replay_buffer.count()))

        state = env.reset()

        episode_reward = 0.
        step = 0
        done = False

        while not done:
            step += 1
            action_clean = agent.model.predict(np.array([state]))

            if is_training:
                epsilon *= EPSILON_DECAY
                epsilon = max(epsilon, EPSILON_MIN)

                action = action_clean + actor_noise() * epsilon
                action = np.clip(action.flatten(), -ACTION_SCALAR, ACTION_SCALAR)
            else:
                action = action_clean

            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten()

            env.render()

            if is_training:
                replay_buffer.add(state, action, reward, next_state, done)      #Add replay buffer

            episode_reward += reward
            state = next_state

            print("Episode", i_episode, "Step", step, "Action", action, "Action clean", action_clean, "Reward", reward, "Epsilon", epsilon)

        if is_training:
            episode_samples = list(replay_buffer.buffer)
            agent.train(episode_samples)
            replay_buffer.erase()

            if i_episode % 5 == 0:
                print("Saving models")
                agent.model.save_weights(agent_path, overwrite=True)

            with open("reward_pend_pg.csv", "a") as reward_file:
                reward_file.write("{};{}\n".format(i_episode, episode_reward))

            print("TOTAL REWARD @ " + str(i_episode) +"-th Episode  : Reward " + str(episode_reward))

            print("Total Step: " + str(step))
            print("")


if __name__ == "__main__":
    main()
