import tensorflow as tf
import numpy as np
from datetime import datetime

now = datetime.now()


class PolicyGradient:
    def __init__(self, n_action: int, n_feature: int, gamma: float = 0.9, lr: float = 0.01,
                 graph: bool = False,sess=None):
        self.n_action = n_action
        self.n_feature = n_feature
        self.gamma = gamma
        self.learning_rate = lr
        self.obs_list = []
        self.act_list = []
        self.reward_list = []
        self._build_net()
        self.prob = tf.nn.softmax(self.network, name='act_prob')
        self.value_per_step = tf.placeholder(tf.float32, [None, ], 'value_list')
        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess=sess
        with tf.variable_scope('score'):
            self.episode_acts = tf.placeholder(tf.int32, [None, ], 'episode_actions')
            scorefunc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.episode_acts,
                                                                       logits=self.network)
            loss = tf.reduce_sum(scorefunc * self.value_per_step)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        if graph:
            self.writer = tf.summary.FileWriter("/home/masterlee/tensorboard/policygradient/" +
                                                now.strftime("%Y%m%d-%H%M%S"), self.sess.graph,
                                                flush_secs=30)

    def _build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_feature], 'state')
        k_initilizer, b_initializer = tf.truncated_normal_initializer(), tf.constant_initializer(0.1)
        layer1 = tf.layers.dense(self.state, 10, tf.nn.relu,
                                 kernel_initializer=k_initilizer,
                                 bias_initializer=b_initializer)
        self.network = tf.layers.dense(layer1, self.n_action,
                                       kernel_initializer=k_initilizer,
                                       bias_initializer=b_initializer)

    def store_transition(self, observation, action, reward, done):
        self.obs_list.append(observation)
        self.act_list.append(action)
        self.reward_list.append(reward)
        if done:
            value_list = self._compute_value()
            self.sess.run(self.train_op, {self.state: self.obs_list,
                                          self.episode_acts: self.act_list,
                                          self.value_per_step: value_list})
            self.obs_list = []
            self.act_list = []
            self.reward_list = []

    def _compute_value(self):
        value_list = np.zeros_like(self.reward_list)
        accumulate_value = 0
        for i in reversed(range(len(self.reward_list))):
            accumulate_value = self.gamma * accumulate_value + self.reward_list[i]
            value_list[i] = accumulate_value

        value_list -= np.mean(value_list)
        value_list /= np.std(value_list)
        return value_list

    def choose_action(self, observation):
        prob = self.prob.eval({self.state: observation[np.newaxis, :]})
        action = np.random.choice(np.arange(self.n_action), p=np.squeeze(prob))
        return action
