from Model.models import Discriminator, Generator
from Dataset import dataset_txt, dataset_mat
from Trainer import trainer_txt, trainer_mat


from Utils.utils import *
from Utils import gpu_info
import os
import numpy as np
import tensorflow as tf
import argparse

seed = 789
np.random.seed(seed)
tf.set_random_seed(seed)


if __name__=='__main__':
    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print(gpus_to_use, free_memory)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  # flickr
    args = parser.parse_args()

    if args.dataset == 'cora':
        config_data = {
                       'feature_file': './Database/cora/features.txt',
                       'graph_file': './Database/cora/edges.txt',
                       'label_file': './Database/cora/group.txt',
                       'negative_ratio': 2,
                       'normalize': True}

        dataset = dataset_txt.Dataset(config_data)

        config_gen = {'gen_dim': [512, dataset.num_feas]}
        config_dis = {'dis_dim': [512],
                      'emb_dim': 100}

        config_trainer = {'noise_dim': 128,
                          'common_z_dim': 64,
                          'attribute_dim': dataset.num_feas,
                          'learning_rate_d': 3e-5,
                          'learning_rate_g': 3e-5,
                          'batch_size': 512,
                          'num_epochs': 150,
                          'gamma': 0.1,
                          'beta1': 0.9,
                          'beta2': 0.999,
                          'weight_decay': 0.0005,
                          'log_path': './Log/cora_{}/'.format(time.strftime("%d-%m-%Y_%H:%M:%S"))}

        os.makedirs(config_trainer['log_path'], exist_ok=True)

        trainer_func = trainer_txt

    elif args.dataset == 'flickr':
        config_data = {'mat_file': './Database/flickr/Flickr.mat',
                       'negative_ratio': 1,
                       'normalize': True}

        dataset = dataset_mat.Dataset(config_data)

        config_gen = {'gen_dim': [512, dataset.num_feas]}
        config_dis = {'dis_dim': [512],
                      'emb_dim': 100}

        config_trainer = {'noise_dim': 128,
                          'common_z_dim': 64,
                          'attribute_dim': dataset.num_feas,
                          'learning_rate_d': 3e-5,
                          'learning_rate_g': 3e-5,
                          'batch_size': 512,
                          'num_epochs': 30,
                          'gamma': 20,
                          'beta1': 0.5,
                          'beta2': 0.999,
                          'weight_decay': 0.0005,
                          'log_path': './Log/flickr_{}/'.format(time.strftime("%d-%m-%Y_%H:%M:%S"))}
        os.makedirs(config_trainer['log_path'], exist_ok=True)

        trainer_func = trainer_mat

    file_name = os.path.basename(__file__)
    logging(file_name, config_trainer['log_path'], verbose=1)

    print(args.dataset)
    for keys, values in config_trainer.items():
        print(keys, values)


    generator = Generator(config_gen)
    discriminator = Discriminator(config_dis)
    trainer = trainer_func.Trainer(generator, discriminator, dataset, config_trainer)
    trainer.train()
    trainer.test()
