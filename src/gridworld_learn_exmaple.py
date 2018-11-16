
import numpy as np
import sys

from gym_unity.envs import UnityEnv

env_name = "../env/GridWorld.x86_64"  # Name of the Unity environment binary to launch
env = UnityEnv(env_name)

# Examine environment parameters
print(str(env))

# Reset the environment
initial_observation = env.reset()


for episode in range(10):
    initial_observation = env.reset()
    done = False
    episode_rewards = 0
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
        episode_rewards += reward
    print("Total reward this episode: {}".format(episode_rewards))

env.close()
