import gym
import signal
import tensorflow as tf
import numpy as np

env_name = ['Reacher-v2', 'Hopper-v2', 'Humanoid-v2']

np.random.seed(1)
tf.set_random_seed(1)
interrupt = False


def termin(signum, frame):
    global interrupt
    interrupt = True


signal.signal(signal.SIGINT, termin)
signal.signal(signal.SIGTERM, termin)


def run(env_name: str, algorithm: str, max_ep, max_ep_step):
    env = gym.make(env_name)
    # env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
    path = (r"saved model/" + env_name[:-3].lower() + "/" +
            algorithm + "/" + env_name[:-3].lower() + ".ckpt")
    if algorithm == 'ddpg':
        from ddpg import DDPG
        agent = DDPG(a_dim, s_dim, a_bound, lr_a=1e-4, lr_c=2e-4, var_decay=.9999,
                     var=0.5, batch_size=128, graph=False, memory_capacity=4000)
        saver = tf.train.Saver()
        saver.restore(agent.sess, path)

        for i in range(max_ep):
            observation = env.reset()
            ep_reward = 0

            for step in range(max_ep_step):
                env.render()
                # False if testing
                act = agent.choose_action(observation, explore=False)

                observation_, reward, done, _ = env.step(act)
                ep_reward += reward

                # commented if eval
                # agent.store_transition(observation, act, reward, observation_)

                if done:
                    print(step, "steps")
                    break

                observation = observation_
            # if i >= 50:
            #     ep_summary = tf.Summary(value=[tf.Summary.Value(tag="ep_reward", simple_value=ep_reward)])
            #     agent.writer.add_summary(ep_summary, i - 50)
            print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %f' % agent.var)
            if interrupt:
                break
        saver.save(agent.sess, path, write_meta_graph=False)

        agent.sess.close()

    elif algorithm == 'ppo':
        from baselines.common import tf_util as U

        def train(num_timesteps=10000):
            from baselines.ppo1 import mlp_policy, pposgd_simple
            U.make_session(num_cpu=1).__enter__()

            def policy_fn(name, ob_space, ac_space):
                return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                            hid_size=64, num_hid_layers=2)

            pposgd_simple.learn(env, policy_fn,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=2048,
                                clip_param=0.2, entcoeff=0.0,
                                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                gamma=0.99, lam=0.95, schedule='linear',
                                )

        train(env)
    env.close()


run(env_name[0], 'ddpg', 1000, 1000)
