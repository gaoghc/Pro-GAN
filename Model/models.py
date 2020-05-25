import tensorflow as tf
from Utils.utils import *

w_init = lambda: tf.random_normal_initializer(stddev=0.02)


class Generator(object):
    def __init__(self, config):
        self.config = config

        self.gen_dim = config['gen_dim']

    def forward(self, x, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as scope:

            h = x
            for layer in range(len(self.gen_dim)):
                h = tf.layers.dense(h, units=self.gen_dim[layer], kernel_initializer=w_init())
                if layer < len(self.gen_dim) - 1:
                    h = tf.nn.relu(h)
                else:
                    h = tf.nn.sigmoid(h)
                # print('gen-{}-{}'.format(layer, h.get_shape()))

        return h


class Discriminator(object):
    def __init__(self, config):
        self.config = config
        self.dis_dim = config['dis_dim']
        self.emb_dim = config['emb_dim']

    def forward(self, x, reuse=False, getter=None):
        with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter) as scope:
            h = x
            for layer in range(len(self.dis_dim)):
                h = tf.layers.dense(h, units=self.dis_dim[layer], kernel_initializer=w_init())
                h = lrelu(h)

                # print('dis-{}-{}'.format(layer, h.get_shape()))

            # ==== discriminator======
            pre_logit = tf.layers.dense(h, units=1, kernel_initializer=w_init())

            # ==== embedding=======
            emb = tf.layers.dense(h, units=self.emb_dim, kernel_initializer=w_init())
            emb = tf.nn.l2_normalize(emb, dim=1)


        return pre_logit, emb





