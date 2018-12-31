import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from datetime import datetime

now = datetime.now()


class DDPG:
    def __init__(self, action_dim, n_state, action_bound, tau: float = 0.01, gamma=0.9,
                 lr_a=0.001, lr_c=0.002, batch_size=128, memory_capacity=10000,
                 var=3, graph=False, var_decay=.9995, var_min=0.01):
        self.n_state = n_state
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.memory = np.zeros([memory_capacity, 2 * n_state + action_dim + 1], dtype=np.float32)
        self.index = 0
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.var = var
        self.var_decay = var_decay
        self.var_min = var_min

        self.state = tf.placeholder(tf.float32, [None, n_state], 'current_state')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.next_state = tf.placeholder(tf.float32, [None, n_state], 'next_state')

        self.actor = self._build_actor(self.state)
        critic = self._build_critic(self.state, self.actor)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic')
        ema = tf.train.ExponentialMovingAverage(1 - tau)
        target_update = [ema.apply(a_params), ema.apply(c_params)]

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        actor_ = self._build_actor(self.next_state, reuse=True, custom_getter=ema_getter)
        critic_ = self._build_critic(self.next_state, actor_, reuse=True, custom_getter=ema_getter)

        a_loss = -tf.reduce_mean(critic)
        self.a_train = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=a_params)
        y = self.reward + gamma * critic_
        loss = tf.losses.mean_squared_error(labels=y, predictions=critic)
        with tf.control_dependencies(target_update):
            self.c_train = tf.train.AdamOptimizer(lr_c).minimize(loss, var_list=c_params)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if graph:
            self.writer = tf.summary.FileWriter(r"/home/masterlee/tensorboard/policygradient/ddpg/" +
                                                now.strftime("%Y%m%d-%H%M%S"), self.sess.graph,
                                                flush_secs=30)
            # tf.summary.scalar('loss', a_loss)
            # self.merge = tf.summary.merge_all()

    def _build_actor(self, state, reuse: bool = None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('actor', reuse=reuse, custom_getter=custom_getter):
            layer1 = tf.layers.dense(state, 256, trainable=trainable)
            layer1 = tc.layers.layer_norm(layer1, center=True, scale=True,
                                          activation_fn=tf.nn.relu)
            layer2 = tf.layers.dense(layer1, 64, trainable=trainable)
            layer2 = tc.layers.layer_norm(layer2, center=True, scale=True,
                                          activation_fn=tf.nn.relu)
            layer3 = tf.layers.dense(layer2, 64, trainable=trainable)
            layer3 = tc.layers.layer_norm(layer3, center=True, scale=True,
                                          activation_fn=tf.nn.relu)
            layer4 = tf.layers.dense(layer3, self.action_dim, trainable=trainable,
                                     activation=tf.nn.tanh)
            return tf.multiply(layer4, self.action_bound, name='scaled_action')
        # trainable = True if reuse is None else False
        # with tf.variable_scope('actor', reuse=reuse, custom_getter=custom_getter):
        #     net = tf.layers.dense(state, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
        #     a = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
        #     return tf.multiply(a, self.action_bound, name='scaled_a')

    @staticmethod
    def _build_critic(state, action, reuse: bool = None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('critic', reuse=reuse, custom_getter=custom_getter):
            network_input = tf.layers.dense(state, 256, name='state_layer', trainable=trainable)
            network_input = tc.layers.layer_norm(network_input, center=True, scale=True,
                                                 activation_fn=tf.nn.relu)
            network_input = tf.concat([network_input, action], axis=-1, name='combined_layer')
            network_input = tf.layers.dense(network_input, 64, trainable=trainable)
            network_input = tc.layers.layer_norm(network_input, center=True, scale=True,
                                                 activation_fn=tf.nn.relu)
            network_input = tf.layers.dense(network_input, 64, trainable=trainable)
            network_input = tc.layers.layer_norm(network_input, center=True, scale=True,
                                                 activation_fn=tf.nn.relu)
            network_output = tf.layers.dense(network_input, 1, trainable=trainable,
                                             kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                              maxval=3e-3))
            return network_output
        # trainable = True if reuse is None else False
        # with tf.variable_scope('critic', reuse=reuse, custom_getter=custom_getter):
        #     n_l1 = 30
        #     w1_s = tf.get_variable('w1_s', [self.n_state, n_l1], trainable=trainable)
        #     w1_a = tf.get_variable('w1_a', [self.action_dim, n_l1], trainable=trainable)
        #     b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
        #     net = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1)
        #     return tf.layers.dense(net, 1, trainable=trainable)

    def store_transition(self, s, a, r, s_):
        transition = np.concatenate((s, a, [r], s_))
        self.memory[self.index % self.memory_capacity] = transition
        self.index += 1
        if self.index > self.memory_capacity:
            self._learn()

    # noinspection SpellCheckingInspection
    def _learn(self):
        indices = np.random.choice(self.memory_capacity, self.batch_size)
        sample = self.memory[indices]
        s = sample[:, :self.n_state]
        a = sample[:, self.n_state:self.n_state + self.action_dim]
        r = sample[:, -self.n_state - 1:-self.n_state]
        s_ = sample[:, -self.n_state:]
        self.sess.run(self.c_train, {self.state: s, self.actor: a,
                                     self.reward: r, self.next_state: s_})
        self.sess.run(self.a_train, {self.state: s})
        # _, merge = self.sess.run([self.a_train, self.merge], {self.state: s})
        self.var *= self.var_decay
        self.var = max(self.var, self.var_min)

        # self.writer.add_summary(merge, self.index - self.memory_capacity)
        # tvars = tf.global_variables('actor/a/bias')
        # tvars_vals = self.sess.run(tvars)
        #
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name, val)
        # input()

    def choose_action(self, s, explore=True):
        action = self.sess.run(self.actor, {self.state: s[np.newaxis, :]})[0]
        if explore:
            action = np.clip(np.random.normal(action, self.var),
                             -self.action_bound, self.action_bound)
        return action


if __name__ == '__main__':
    import gym
    import signal

    interrupt = False
    max_ep = 200
    max_ep_step = 200


    def termin(signum, frame):
        global interrupt
        interrupt = True


    signal.signal(signal.SIGINT, termin)
    signal.signal(signal.SIGTERM, termin)
    env = gym.make('Humanoid-v2')
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    agent = DDPG(a_dim, s_dim, a_bound, lr_a=0.001, lr_c=0.002, var_decay=.9995,
                 graph=False)

    saver = tf.train.Saver()
    # save_path = saver.restore(agent.sess, r"saved model/pendulum/ddpg/pendulum_ddpg.ckpt")

    for i in range(max_ep):
        observation = env.reset()
        ep_reward = 0

        for step in range(max_ep_step):
            # if i>70:
            #     env.render()
            # False if testing
            act = agent.choose_action(observation, explore=True)

            observation_, reward, _, _ = env.step(act)

            reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0
            # so when Q at this state is greater than 0, the agent overestimates the Q.
            ep_reward += reward * 10
            agent.store_transition(observation, act, reward, observation_)

            observation = observation_
        if i >= 50:
            ep_summary = tf.Summary(value=[tf.Summary.Value(tag="ep_reward", simple_value=ep_reward)])
            # agent.writer.add_summary(ep_summary, i - 50)
        print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %f' % agent.var)
        if interrupt:
            break
    save_path = saver.save(agent.sess, r"saved model/pendulum/ddpg/pendulum_ddpg.ckpt")
