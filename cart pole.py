#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:11:03 2018

@author: masterlee
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dqn import DeepQNetwork
from gym import wrappers
from policygradient import PolicyGradient
from datetime import datetime
env = gym.make('CartPole-v0')
env = env.unwrapped
env = wrappers.Monitor(env, r'/home/masterlee/reinforcement learning/video/cart pole/'
                       + datetime.now().strftime("%Y%m%d-%H%M%S"), lambda x: x>30,
                       force=True,resume=False)
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

tf.reset_default_graph()
agent = DeepQNetwork(env.action_space.n, env.observation_space.shape[0], double=False,
                     polyak_averaging=False, memory_size=2000, lr=0.01, pre_train=False,
                     replace_iter=100, priority=False, learn_threshold=500, dueling=True)

# agent = PolicyGradient(env.action_space.n, env.observation_space.shape[0], lr=0.02, gamma=0.9)
# saver = tf.train.Saver()
# save_path = saver.restore(agent.sess, r"saved model/cart pole/dqn/cart_pole_dqn.ckpt")

# if dont want to train,set this
agent.learn_threshold = 1e8


reward_per_episode = np.zeros(100)
episode = 0
observation = env.reset()
reward_sum = 0
reward_list = []
while reward_per_episode.mean() < 195:
    if episode > 150:
        env.render()

    # false means act deterministicly
    action = agent.choose_action(observation)
    # print(action, end=' ')

    next_observation, reward, done, _ = env.step(action)
    reward_sum += 1
    # dqn store
    x, x_dot, theta, theta_dot = next_observation
    r1 = (env.env.x_threshold - abs(x)) / env.env.x_threshold - 0.8
    r2 = (env.env.theta_threshold_radians - abs(theta)) / env.env.theta_threshold_radians - 0.5
    reward = r1 + r2
    agent.store_transition(observation, action, reward, next_observation, done)

    # policy gradient store
    # agent.store_transition(observation, action, reward, done)

    observation = next_observation
    if done:
        print("")
        print("episode", episode, ":", reward_sum, "steps")
        reward_per_episode[episode % 100] = reward_sum
        reward_list.append(reward_sum)
        # step_summary = tf.Summary(value=[tf.Summary.Value(tag="step", simple_value=reward_sum)])
        # agent.writer.add_summary(step_summary, episode)
        episode += 1
        reward_sum = 0
        observation = env.reset()
        # break
env.close()
# env.env.close() #close renderer
# agent.plot_cost()

plt.plot(reward_list)
plt.show()
# saver = tf.train.Saver()
# save_path = saver.save(agent.sess, r"saved model/cart pole/dqn/cart_pole_dqn.ckpt")
# print(save_path)

agent.sess.close()
