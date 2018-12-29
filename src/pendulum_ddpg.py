"""
Implementation of the DDPG algorithm
(see "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING" - Lillicrap et al.)
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
    '''
    Class to create explorative noise. An Ornstein-Uhlenbeck-Process is used in this case,
    because it provides better exploration than gaussian noise.
    '''
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
    '''
    Stores all the simulated trajectories. And provides functions to randomly sample experience from the buffer.
    '''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
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
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class Actor:
    '''
    Encapsulation of the actor model.
    '''
    def __init__(self, session, lr, tau, action_space_size, state_space_size, action_scalar=1):
        self.learning_rate = lr
        self.tau = tau
        self.sess = session

        self.action_scalar = action_scalar

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        self.state_input, self.model = self.create_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, action_space_size])

        model_weights = self.model.trainable_weights
        self.actor_grads = tf.gradients(self.model.output, model_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        _, self.target_model = self.create_model()

    def create_model(self):
        '''
        Function to build the model
        '''

        # uniform weight initialization with very small weights provided the best test results
        initializer = RandomUniform(-0.005, 0.005)

        state_input = Input(shape=[self.state_space_size])
        h1 = Dense(128, activation='relu', kernel_initializer=initializer)(state_input)
        h2 = Dense(128, activation='relu', kernel_initializer=initializer)(h1)
        h3 = Dense(64, activation='relu', kernel_initializer=initializer)(h2)
        output = Dense(self.action_space_size, activation='tanh', kernel_initializer=initializer)(h3)
        scaled_output = Lambda(lambda x: x * self.action_scalar)(output)

        model = Model(input=state_input, output=scaled_output)
        adam = Adam(self.learning_rate)
        model.compile(loss=MSE, optimizer=adam)
        return state_input, model

    def train(self, samples, critic_grads):
        sample_grad_pairs = zip(samples, critic_grads)

        for sample, c_grad in sample_grad_pairs:
            cur_state, _, _, _, _ = sample

            cur_state = np.array([cur_state])
            c_grad = np.array([c_grad])

            self.sess.run(self.optimize, feed_dict={
                self.state_input: cur_state,
                self.actor_critic_grad: c_grad
            })

    def update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)


class Critic:
    '''
    Encapsulation of the critic model
    '''
    def __init__(self, session, lr, tau, action_space_size, state_space_size):
        self.sess = session

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        self.learning_rate = lr
        self.tau = tau

        self.state_input, self.action_input, self.model = self.create_model()
        self.action_grads = tf.gradients(self.model.output, self.action_input)

        _, _, self.target_model = self.create_model()

    def create_model(self):
        '''
        function to build the model
        '''
        initializer = RandomUniform(-0.05, 0.05)

        state_input = Input(shape=[self.state_space_size])
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(64)(state_h1)

        action_input = Input(shape=[self.action_space_size])
        action_h1    = Dense(64)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear', kernel_initializer=initializer)(merged_h1)
        model  = Model(input=[state_input, action_input], output=output)

        adam  = Adam(lr=self.learning_rate)
        model.compile(loss=MSE, optimizer=adam)
        return state_input, action_input, model

    def gradients(self, states, actions):
        '''
        calculate gradients of the critic model

        :param states:
        :param actions:
        :return:
        '''
        return self.sess.run(self.action_grads, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })[0]

    def train(self, samples, target_qs):
        assert len(samples) == len(target_qs)
        y_target = np.zeros(len(samples))

        for i in range(len(samples)):
            _, _, reward, _, done = samples[i]
            y_target[i] = reward if done else reward + target_qs[i]

        states = np.asarray([e[0] for e in samples])
        actions = np.asarray([e[1] for e in samples])

        self.model.fit([states, actions], y_target)

    def update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)


def main():
    # Hyperparameters
    BUFFER_SIZE = 20000
    EPSILON_DECAY = 0.9999
    EPSILON_MIN = 0.05
    BATCH_SIZE = 64
    LR_ACTOR = 0.0001
    LR_CRITIC = 0.001
    TAU = 0.001
    ACTION_SCALAR = 2

    # path to load/safe the trained models
    actor_path = "actor_pend_a2c.h5"
    critic_path = "critic_pend_a2c.h5"

    is_training = False

    try:
        is_training = sys.argv[1] == "train"
    except:
        pass

    # create the environment
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    K.set_session(sess)

    # create actor and critic
    actor = Actor(sess, LR_ACTOR, TAU, action_dim, state_dim, action_scalar=ACTION_SCALAR)
    critic = Critic(sess, LR_CRITIC, TAU, action_dim, state_dim)

    print("TRAINING" if is_training else "RUN")

    if not is_training:
        actor.load_weights(actor_path)
        critic.load_weights(critic_path)


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

            # the amount of noise that is added to the action is decaying over time
            epsilon *= EPSILON_DECAY
            epsilon = max(epsilon, EPSILON_MIN)

            action_clean = actor.model.predict(np.array([state]))

            if is_training:
                action = action_clean + actor_noise() * epsilon
                action = np.clip(action.flatten(), -ACTION_SCALAR, ACTION_SCALAR)
            else:
                action = action_clean

            # retrieve next state and reward from the environment
            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten()

            env.render()

            if is_training:
                # add experience to the buffer
                replay_buffer.add(state, action, reward, next_state, done)      #Add replay buffer
                # take random batch of experience from the buffer
                batch = replay_buffer.get_batch(BATCH_SIZE)

                if len(batch) == BATCH_SIZE:
                    states = np.asarray([e[0] for e in batch])
                    next_states = np.asarray([e[3] for e in batch])

                    # calculate target q-values from the target models
                    target_q_values = critic.target_model.predict([next_states,
                                                                   actor.target_model.predict(next_states)])
                    # train the critic (simple supervised learning)
                    critic.train(batch, target_q_values)
                    # get actions for gradient calculation
                    a_for_grad = actor.model.predict(states)
                    # get critic gradients
                    grads = critic.gradients(states, a_for_grad)
                    # train actor
                    actor.train(batch, grads)
                    # update the target models
                    actor.update_target_model()
                    critic.update_target_model()

            episode_reward += reward
            state = next_state

            print("Episode", i_episode, "Step", step, "Action", action, "Action clean", action_clean, "Reward", reward, "Epsilon", epsilon)

        if is_training:
            # safe model and print training states

            if i_episode % 5 == 0:
                print("Saving models")
                actor.model.save_weights(actor_path, overwrite=True)
                critic.model.save_weights(critic_path, overwrite=True)

            with open("reward_pend_a2c.csv", "a") as reward_file:
                reward_file.write("{};{}\n".format(i_episode, episode_reward))

            print("TOTAL REWARD @ " + str(i_episode) +"-th Episode  : Reward " + str(episode_reward))

            print("Total Step: " + str(step))
            print("")


if __name__ == "__main__":
    main()
