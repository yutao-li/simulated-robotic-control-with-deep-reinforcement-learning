import gym
from dqn import DeepQNetwork
import numpy as np
import tensorflow as tf
from datetime import datetime
from gym import wrappers

env = gym.make('Pendulum-v0')
env = env.unwrapped
env = wrappers.Monitor(env, r'/home/masterlee/reinforcement learning/video/pendulum/'
                       + datetime.now().strftime("%Y%m%d-%H%M%S"), lambda x: True, force=True, resume=False)
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25

agent = DeepQNetwork(n_action=ACTION_SPACE, n_feature=3, memory_size=MEMORY_SIZE,
                     double=True, lr=0.005, replace_iter=200, dueling=True, learn_threshold=500)

saver = tf.train.Saver()
save_path = saver.restore(agent.sess, r"saved model/pendulum/dqn/pendulum_dqn.ckpt")

# if dont want to train,set this
agent.learn_threshold = 1e8


def train(agent):
    total_steps = 0
    observation = env.reset()
    while True:
        # env.render()

        # false means act deterministicly
        action = agent.choose_action(observation, False)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        agent.store_transition(observation, action, reward, observation_, done)

        # adjusted for video recorder or training
        threshold = 1000
        if total_steps > threshold:  # stop game
            break

        observation = observation_
        total_steps += 1
        if total_steps % 1000 == 0:
            print(total_steps, "steps")


train(agent)
env.close()
env.env.close()
saver = tf.train.Saver()
# save_path = saver.save(agent.sess, r"saved model/pendulum/dqn/pendulum_dqn.ckpt")
agent.sess.close()
