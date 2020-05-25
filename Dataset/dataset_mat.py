import numpy as np
import scipy.sparse as sp
import warnings
import linecache
from sklearn import preprocessing
import os
import scipy.io as sio
from sklearn.feature_extraction.text import TfidfTransformer
import random
import math


class Dataset(object):

    def __init__(self, config):
        self.config = config
        self.mat_file = config['mat_file']
        self.normalize = config['normalize']
        self.negative_ratio = config['negative_ratio']

        self.W, self.X, self.Y = self._load_dataset_mat()

        if self.normalize:
            self.X = preprocessing.normalize(self.X, norm='l2')

        self.num_nodes = self.X.shape[0]
        self.num_edges = np.int32(np.sum(self.W))
        self.num_feas = self.X.shape[1]


        self._gen_sampling_table()

    def _load_dataset_mat(self):
        data = sio.loadmat(self.mat_file)

        W = data['Network']
        Y = data['Label']
        Z = data['Attributes']

        num_nodes = Y.shape[0]
        num_classes = len(np.unique(Y))
        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for id in range(num_nodes):
            L[id, Y[id][0] - 1] = 1

        W = W.tocoo()
        Z = Z.todense()


        return W, Z, L

    def _gen_sampling_table(self):
        table_size = 1e8
        power = 0.75

        numNodes = self.num_nodes
        node_degree = np.zeros(numNodes)  # out degree

        for row, col in zip(self.W.row, self.W.col):
            node_degree[row] += 1

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.num_edges
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = self.num_edges
        norm_prob = [data_size / total_sum for _ in range(self.num_edges)]

        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size - 1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + \
                                         norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1

    def batch_iter(self, batch_size):
        table_size = 1e8
        negative_ratio = self.negative_ratio
        numNodes = self.num_nodes

        edges = [(row, col) for row, col in zip(self.W.row, self.W.col)]
        data_size = self.num_edges
        edge_set = set([x[0] * numNodes + x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))

        mod = 0
        mod_size = 1 + negative_ratio
        h = []
        t = []

        sign = 0

        start_index = 0
        end_index = min(start_index + batch_size, data_size)
        while start_index < data_size:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    t.append(
                        self.sampling_table[random.randint(0, table_size - 1)])

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + batch_size, data_size)







