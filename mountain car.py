import gym
from dqn import DeepQNetwork
import matplotlib.pyplot as plt
from gym import wrappers
import tensorflow as tf
from datetime import datetime

env = gym.make('MountainCar-v0')
env = env.unwrapped
env = wrappers.Monitor(env, r'/home/masterlee/reinforcement learning/video/mountain car/'
                       + datetime.now().strftime("%Y%m%d-%H%M%S"),
                       lambda x: x > 15, force=True, resume=False)
env.seed(21)
MEMORY_SIZE = 10000
agent = DeepQNetwork(3, 2, memory_size=MEMORY_SIZE, replace_iter=500, priority=True, double=True,
                     dueling=True, exploring_rate=0.0001, learn_threshold=5000)
saver = tf.train.Saver()
save_path = saver.restore(agent.sess, r"saved model/mountain car/dqn/mountain car_dqn.ckpt")

# if dont want to train,set this
# agent.learn_threshold = 1e8

steps = []

for i_episode in range(20):
    total_steps = 0
    observation = env.reset()
    while True:
        # if i_episode > 10:
        #     env.render()

        # false means act deterministicly
        action = agent.choose_action(observation, True)

        observation_, reward, done, info = env.step(action)

        if done:
            reward = 10

        agent.store_transition(observation, action, reward, observation_, done)

        if done:
            print('episode', i_episode, total_steps)
            steps.append(total_steps)
            break

        observation = observation_
        total_steps += 1

    # used just for recorder
    # break

plt.plot(steps)
plt.show()
saver = tf.train.Saver()
save_path = saver.save(agent.sess, r"saved model/mountain car/dqn/mountain car_dqn.ckpt")

agent.sess.close()
env.close()
env.env.close()  # close renderer
