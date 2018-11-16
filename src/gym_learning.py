import numpy as np
import sys

import gym

from keras.layers import Dense, Conv2D, Flatten, Activation
from keras import  Sequential
from keras.optimizers import Adam
from keras.initializers import normal
from keras.models import load_model
from keras.callbacks import TensorBoard

from collections import namedtuple, deque

import random

from gym_unity.envs import UnityEnv

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        #self.epsilon = 1.0
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .25

        #self.model = load_model("./trial-990.model")
        #self.target_model = load_model("./trial-990.model")

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.tb_cb = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                 write_graph=True, write_images=True)

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, input_shape=env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(self.env.action_space.n))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 20
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


env = gym.make('CartPole-v0')

episodes = 2000

# updateTargetNetwork = 1000
dqn_agent = DQN(env=env)
steps = []
env.reset()

for ep in range(episodes):
    cur_state = np.array([env.reset()])
    done = False
    ep_reward = 0
    n_steps = 0
    while not done:
        action = dqn_agent.act(cur_state)
        new_state, reward, done, info = env.step(action)
        new_state = np.array([new_state])

        ep_reward += reward

        # reward = reward if not done else -20
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        dqn_agent.replay()       # internally iterates default (prediction) model
        dqn_agent.target_train() # iterates target model

        cur_state = new_state
        n_steps += 1
        env.render()

    print("Total reward: {}  Steps: {}".format(ep_reward,n_steps))
    if ep % 10 == 0:
        dqn_agent.save_model("si-trial-{}.model".format(ep))

env.close()
