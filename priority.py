import numpy as np


class SumTree:
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, length):
        self.capacity = capacity
        self.priority_tree = np.zeros(capacity * 2 - 1, dtype=np.float32)
        self.data = np.zeros((capacity, length), dtype=np.float32)
        self.data_index = 0
        self.full = False

    def add(self, transition):
        self.data[self.data_index] = transition
        priority = np.max(self.priority_tree[-self.capacity:])
        if priority == 0:
            priority = SumTree.abs_err_upper
        self.update(self.data_index + self.capacity - 1, priority)
        self.data_index += 1
        if self.data_index == self.capacity:
            self.full = True
            self.data_index = 0

    def update(self, tree_idx, priority):
        change = priority - self.priority_tree[tree_idx]
        while tree_idx != 0:
            self.priority_tree[tree_idx] += change
            tree_idx = (tree_idx - 1) // 2
        self.priority_tree[0] += change

    def sample(self, priority):
        parent = 0
        while True:
            lchild = parent * 2 + 1
            rchild = parent * 2 + 2
            if lchild >= self.capacity * 2 - 1:
                leaf = parent
                break
            else:
                if priority > self.priority_tree[lchild]:
                    priority -= self.priority_tree[lchild]
                    parent = rchild
                else:
                    parent = lchild
        return leaf, self.priority_tree[leaf], self.data[leaf - self.capacity + 1]

    def nsample(self, n):
        batch_tree_idx = np.zeros(n, dtype=np.int32)
        batch_data = np.zeros((n, self.data.shape[1]))
        batch_weight = np.zeros(n)
        prior_seg = self.priority_tree[0] / n
        self.beta = np.minimum(1., self.beta + SumTree.beta_increment_per_sampling)
        if self.full:
            minp = np.min(self.priority_tree[-self.capacity:])
        else:
            minp = np.min(self.priority_tree[-self.capacity:self.data_index - self.capacity])
        for i in range(n):
            a, b = prior_seg * i, prior_seg * (i + 1)
            prior_sample = np.random.uniform(a, b)
            idx, priority, data = self.sample(prior_sample)
            batch_tree_idx[i] = idx
            batch_data[i] = data
            batch_weight[i] = (priority / minp) ** (-self.beta)
        return batch_tree_idx, batch_data, batch_weight

    def batch_update(self, batch_idx, batch_diff):
        batch_diff = np.minimum(SumTree.abs_err_upper, batch_diff)  # clipped
        batch_diff = np.power(batch_diff + SumTree.epsilon, SumTree.alpha)
        for idx, diff in zip(batch_idx, batch_diff):
            self.update(idx, diff)
