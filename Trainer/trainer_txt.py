import tensorflow as tf
import numpy as np
import time
from Utils.utils import *


class Trainer(object):

    def __init__(self, generator, discriminator, dataset, config):
        self.config = config
        self.noise_dim = config['noise_dim']
        self.attribute_dim = config['attribute_dim']
        self.learning_rate_d = config['learning_rate_d']
        self.learning_rate_g = config['learning_rate_g']
        self.batch_size = config['batch_size']
        self.common_z_dim = config['common_z_dim']
        self.num_epochs = config['num_epochs']
        self.gamma = config['gamma']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.log_path = config['log_path']
        self.weight_decay = config['weight_decay']


        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset

        self.z_anchor = tf.placeholder(tf.float32, [None, self.noise_dim])
        self.z_pos = tf.placeholder(tf.float32, [None, self.noise_dim])
        self.z_neg = tf.placeholder(tf.float32, [None, self.noise_dim])
        self.lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')

        self.x_anchor = tf.placeholder(tf.float32, [None, self.attribute_dim])
        self.x_context = tf.placeholder(tf.float32, [None, self.attribute_dim])
        self.sign = tf.placeholder(tf.float32, [None])

        self.loss_dis, self.loss_gen, self.opt_dis, self.opt_gen, self.loss_d, self.loss_dist_real, self.loss_dist_fake = self._build_training_graph()
        self.mu = self._build_testing_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def _build_training_graph(self):
        x_anchor_fake = self.generator.forward(self.z_anchor, reuse=False)
        x_pos_fake = self.generator.forward(self.z_pos, reuse=True)
        x_neg_fake = self.generator.forward(self.z_neg, reuse=True)


        pre_logit_anchor_real, mu_anchor_real = self.discriminator.forward(self.x_anchor, reuse=False)
        pre_logit_context, mu_context = self.discriminator.forward(self.x_context, reuse=True)


        pre_logit_anchor_fake, mu_anchor_fake = self.discriminator.forward(x_anchor_fake, reuse=True)
        pre_logit_pos_fake, mu_pos_fake = self.discriminator.forward(x_pos_fake, reuse=True)
        pre_logit_neg_fake, mu_neg_fake = self.discriminator.forward(x_neg_fake, reuse=True)

        real = tf.concat((pre_logit_anchor_real, pre_logit_context), axis=0)
        fake = tf.concat((pre_logit_anchor_fake, pre_logit_pos_fake, pre_logit_neg_fake), axis=0)
        loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        loss_d = loss_d_real + loss_d_fake

        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))

        sim = tf.reduce_sum(tf.multiply(mu_anchor_real, mu_context), 1)
        loss_dist_real = -tf.reduce_mean(tf.log_sigmoid(self.sign * sim))

        sim_anchor_pos_fake = tf.reduce_sum(tf.multiply(mu_anchor_fake, mu_pos_fake), 1)
        sim_anchor_neg_fake = tf.reduce_sum(tf.multiply(mu_anchor_fake, mu_neg_fake), 1)
        loss_anchor_pos_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(sim_anchor_pos_fake),
                                                                                 logits=sim_anchor_pos_fake))
        loss_anchor_neg_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sim_anchor_neg_fake),
                                                                                 logits=sim_anchor_neg_fake))

        loss_dist_fake = loss_anchor_pos_fake + loss_anchor_neg_fake


        loss_dis = loss_d + loss_dist_real + loss_dist_fake * self.gamma
        loss_gen = loss_g + loss_dist_real + loss_dist_fake


        vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        weight_decay_dis = tf.add_n([tf.nn.l2_loss(v) for v in vars_dis if 'bias' not in v.name]) * self.weight_decay

        opt_gen = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1, beta2=self.beta2).minimize(loss_gen, var_list=vars_gen)
        opt_dis = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1, beta2=self.beta2).minimize(loss_dis+weight_decay_dis, var_list=vars_dis)



        return loss_dis, loss_gen, opt_dis, opt_gen, loss_d, loss_dist_real, loss_dist_fake

    def _build_testing_graph(self):
        _, emb = self.discriminator.forward(self.x_anchor, reuse=True)

        return emb


    def train(self):

        spec_dim = self.noise_dim - self.common_z_dim

        for epoch in range(self.num_epochs):


            batches = self.dataset.batch_iter(self.batch_size)

            cnt = 0
            loss_d = 0
            loss_g = 0
            loss_r = 0
            start_time = time.time()
            for batch in batches:
                h, t, sign = batch
                hx = self.dataset.X[h]
                ht = self.dataset.X[t]

                cur_size = hx.shape[0]
                z_common = np.random.uniform(-1, 1, size=(cur_size, self.common_z_dim))
                z_speficic_1 = np.random.uniform(-1, 1, size=(cur_size, spec_dim))
                z_speficic_2 = np.random.uniform(-1, 1, size=(cur_size, spec_dim))
                z_neg = np.random.uniform(-1, 1, size=(cur_size, self.noise_dim))
                z_pos = np.concatenate((z_common, z_speficic_2), axis=1)
                z_anchor = np.concatenate((z_common, z_speficic_1), axis=1)

                _, cost_dis_batch, cost_real = self.sess.run([self.opt_dis, self.loss_d, self.loss_dist_real],
                                                             feed_dict={self.x_anchor: hx,
                                                                        self.x_context: ht,
                                                                        self.sign: sign,
                                                                        self.z_anchor: z_anchor,
                                                                        self.z_pos: z_pos,
                                                                        self.z_neg: z_neg})

                _, cost_gen_batch = self.sess.run([self.opt_gen, self.loss_gen],
                                                  feed_dict={self.x_anchor: hx,
                                                             self.x_context: ht,
                                                             self.sign: sign,
                                                             self.z_anchor: z_anchor,
                                                             self.z_pos: z_pos,
                                                             self.z_neg: z_neg})


                loss_d += cost_dis_batch
                loss_g += cost_gen_batch
                loss_r += cost_real
                cnt += 1
            end_time = time.time()
            loss_d /= cnt
            loss_g /= cnt
            loss_r /= cnt


            print('Epoch {}: loss_d {:.4f}, loss_g {:.4f}, loss_r {:.4f}, '
                  'time {:.4f}'.format(epoch, loss_d, loss_g, loss_r,end_time - start_time))


        self.saver.save(self.sess, os.path.join(self.log_path, 'model_final.pkl'))


    def test(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(self.log_path, 'model_final.pkl'))

        num_val = self.dataset.num_nodes
        perm = np.arange(num_val)

        E = []
        index = 0
        while index < num_val:
            if index + self.batch_size < num_val:
                perm_ = perm[index:index + self.batch_size]
            else:
                perm_ = perm[index:]
            mini_batch_att = self.dataset.X[perm_]
            index += self.batch_size

            emb = self.sess.run(self.mu, feed_dict={self.x_anchor: mini_batch_att})
            E.extend(emb)
        E = np.array(E)


        test_ratio = np.arange(0.5, 1.0, 0.2)
        res = []
        for tr in test_ratio[-1::-1]:
            print('============train ratio-{}=========='.format(1 - tr))
            micro, macro = multi_label_classification(E, self.dataset.Y, tr)
            res.append('{:.4f}'.format(micro) + ' ' + '{:.4f}'.format(macro))
        print(' '.join(res))



