#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:36:13 2018

@author: masterlee
"""

import math
import numpy as np
import tensorflow as tf
from datetime import datetime
from priority import SumTree

np.random.seed(1)
tf.set_random_seed(1)
now = datetime.now()


class DeepQNetwork:
    def __init__(self, n_action: int, n_feature: int, double: bool = False,
                 polyak_averaging: bool = False, lr: float = 0.01, sess=None,
                 memory_size: int = 10000, replace_iter: int = 300,
                 pre_train: bool = False, priority: bool = True,
                 exploring_rate=0.002, learn_threshold: int = 1000,
                 dueling: bool = False, graph: bool = False):
        self.n_action = n_action
        self.n_feature = n_feature
        self.memory_size = memory_size
        self.gamma = 0.9
        self.batch_size = 32
        self.epsilon = 1
        self.learning_step = 1
        self.learning_rate = lr
        self.priority = priority
        self.memory_counter = 0
        self.exploring_rate = exploring_rate
        self.learn_threshold = learn_threshold
        self.dueling = dueling
        if priority:
            self.memory = SumTree(memory_size, n_feature * 2 + 3)
        else:
            self.memory = np.zeros((self.memory_size, n_feature * 2 + 3), dtype=np.float32)
        self.double = double
        self.polyak_averaging = polyak_averaging
        self.replace_iter = replace_iter
        self.pre_train = pre_train
        self._build_net()
        self.eval_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        self.polyak_averaging_op = [tf.assign(a, 0.999 * a + 0.001 * b) for a, b
                                    in zip(self.target_net_params, self.eval_net_params)]
        self.replace_params_op = [tf.assign(a, b) for a, b
                                  in zip(self.target_net_params, self.eval_net_params)]

        self.eval_for_next_state = tf.placeholder(tf.float32, [self.batch_size, self.n_action],
                                                  'eval_for_next_state')
        with tf.variable_scope('loss'):
            indices = tf.stack([tf.range(self.batch_size), self.action], axis=1)
            selected_action_eval = tf.gather_nd(self.q_eval, indices)
            if not self.double:
                y = self.reward + self.gamma * (1 - self.done) * tf.reduce_max(self.q_target, axis=1)
                self.y_target = tf.stop_gradient(y, name='fixed_target')
            else:
                eval_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                         tf.argmax(self.eval_for_next_state, output_type=tf.int32, axis=1)],
                                        axis=1)
                y = self.reward + self.gamma * (1 - self.done) * tf.gather_nd(self.q_target, eval_indices)
                self.y_target = tf.stop_gradient(y, name='fixed_target')

            if self.priority:
                self.importance_weight = tf.placeholder(tf.float32, [self.batch_size, ], 'ISweights')
                self.abs_error = tf.abs(self.y_target - selected_action_eval, 'abs_error')
                self.loss = tf.losses.mean_squared_error(self.y_target, selected_action_eval,
                                                         self.importance_weight)
            else:
                self.loss = tf.losses.mean_squared_error(self.y_target, selected_action_eval)
            tf.summary.scalar('loss', self.loss)

        # used before fixed target training to train the variables a bit
        # with tf.variable_scope('pre_train'):
        #     eval_next = self.reward + self.gamma * (1 - self.done) * tf.reduce_max(self.eval_for_next_state, axis=1)
        #     loss = tf.losses.mean_squared_error(eval_next, selected_action_eval)
        #     self.pre_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        with tf.variable_scope('train'):

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        self.sess.run(self.replace_params_op)
        self.loss_record = []
        if graph:
            self.writer = tf.summary.FileWriter("/home/masterlee/tensorboard/dqn/" +
                                                now.strftime("%Y%m%d-%H%M%S"), self.sess.graph,
                                                flush_secs=30)
        self.merge_op = tf.summary.merge_all()

    def _build_net(self):
        self.state = tf.placeholder(tf.float32, shape=[None, self.n_feature],
                                    name='state')
        self.action = tf.placeholder(tf.int32, shape=[None, ], name='action')
        self.reward = tf.placeholder(tf.float32, shape=[None, ], name='reward')
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.n_feature],
                                         name='next_state')
        self.done = tf.placeholder(tf.float32, [None, ], 'done')

        def build(network_input, name: str, neuron: int = 10) -> tf.Tensor:
            if name == 'eval':
                w_initializer, b_initializer = [tf.truncated_normal_initializer(),
                                                tf.constant_initializer(0.1)]
            else:
                w_initializer, b_initializer = [tf.zeros_initializer(),
                                                tf.zeros_initializer()]
            layer1 = tf.layers.dense(network_input, neuron, activation=tf.nn.relu, name=name + '_layer1',
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer)
            if self.dueling:
                value = tf.layers.dense(layer1, 1, name=name + '_value',
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer)
                advantage = tf.layers.dense(layer1, self.n_action, name=name + 'advantage',
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer)
                layer2 = advantage - tf.reduce_mean(advantage, 1, True) + value
            else:
                layer2 = tf.layers.dense(layer1, self.n_action, name=name + '_layer2',
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer)
            return layer2

        '''
        ---------eval_net----------
        '''
        with tf.variable_scope('eval_net'):
            self.q_eval = build(self.state, 'eval')
            tf.summary.histogram("eval_net", self.q_eval)
        '''
        ----------target_net----------
        '''
        with tf.variable_scope('target_net'):
            self.q_target = build(self.next_state, 'target')
            tf.summary.histogram('target_net', self.q_target)

    def choose_action(self, observation, greedy: bool = True):
        observation = observation[np.newaxis, :]
        action = np.argmax(self.sess.run(
            self.q_eval, feed_dict={self.state: observation}))
        if greedy:
            if np.random.rand() < 1 - self.epsilon:
                return action
            else:
                return np.random.randint(self.n_action)
        else:
            return action

    def _learn(self):
        if self.priority:
            tree_idx, batch, is_weights = self.memory.nsample(self.batch_size)
        else:
            if self.memory_counter < self.memory_size:
                sample_index = np.random.choice(self.memory_counter, self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_size, self.batch_size)
            batch = self.memory[sample_index]
        state, action, reward, next_state, done = [batch[:, :self.n_feature],
                                                   batch[:, self.n_feature],
                                                   batch[:, self.n_feature + 1],
                                                   batch[:, -1 - self.n_feature:-1],
                                                   batch[:, -1]]
        # if self.pre_train:
        #     if self.learning_step < 20:
        #         eval_for_next_state = self.q_eval.eval({self.state: next_state})
        #         self.sess.run(self.pre_train_op, {self.state: state,
        #                                           self.action: action,
        #                                           self.reward: reward,
        #                                           self.done: done,
        #                                           self.eval_for_next_state: eval_for_next_state})
        #         self.learning_step += 1
        #         self.epsilon_k = 0.01 + 0.99 * math.exp(-.002 * self.learning_step)
        #         return
        #     elif self.learning_step == 20:
        #         self.sess.run(self.replace_params_op)
        #     else:
        #         pass

        feed_dict = {self.next_state: next_state,
                     self.state: state,
                     self.action: action,
                     self.reward: reward,
                     self.done: done}
        if self.priority:
            feed_dict[self.importance_weight] = is_weights
        if self.double:
            q_eval_next = self.sess.run(self.q_eval, {self.state: next_state})
            feed_dict[self.eval_for_next_state] = q_eval_next

        loss, merge, _ = self.sess.run([self.loss, self.merge_op, self.train_op],
                                       feed_dict)
        if self.priority:
            abs_error = self.sess.run(self.abs_error, feed_dict)
            self.memory.batch_update(tree_idx, abs_error)
        self.loss_record.append(loss)
        # self.writer.add_summary(merge, self.learning_step)
        self.learning_step += 1
        self.epsilon = 0.01 + 0.99 * math.exp(-self.exploring_rate * self.learning_step)
        if self.polyak_averaging:
            self.sess.run(self.polyak_averaging_op)
        elif self.learning_step % self.replace_iter == 0:
            print("\nreplace")
            self.sess.run(self.replace_params_op)

    def store_transition(self, observation, action, reward, next_obs, done):
        transition = np.hstack((observation, action, reward, next_obs, done))
        if self.priority:
            self.memory.add(transition)
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index] = transition
        self.memory_counter += 1
        if self.memory_counter > self.learn_threshold:
            self._learn()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_record)
        plt.show()
