import click
import torch
import logging
import warnings

import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'mvtecad']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'mvtecad_LeNet_ELU']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,help='Config JSON-file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=[0], multiple=True, help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--discount_factor', type=float, default=0.0,help='Discount factor with centor moving rate.')
@click.option('--eps', type=bool, default=False,help='Epsilon-greedy based on centor moving rate.')
@click.option('--pretrain', type=bool, default=True,help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=[0], multiple=True,help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0, help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, device,
         seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, discount_factor, eps,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # # Turn off warning messages
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)

    # Print arguments
    logger.info('Data import path is %s.' % data_path)
    logger.info('Data export path is %s.' % xp_path)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed
    if cfg.settings['seed'] == -1:
        cfg.settings['seed'] = random.randint(0, 2**32 - 1)
    random.seed(cfg.settings['seed'])
    np.random.seed(cfg.settings['seed'])
    torch.manual_seed(cfg.settings['seed'])
    logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    if 'mvtecad_LeNet_ELU' == net_name:
        net_name = net_name + '-' + str(normal_class)
    deep_SVDD = DeepSVDD(); 
    deep_SVDD.set_network(net_name)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)
        torch.save(deep_SVDD.net.state_dict(), f'{xp_path}/latent-{str(normal_class)}.pth')
    else:
        deep_SVDD.net.load_state_dict(torch.load(f'{xp_path}/latent-{str(normal_class)}.pth'))

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    discount_factor=cfg.settings['discount_factor'],
                    eps=cfg.settings['eps'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    outputs, labels, scores = zip(*deep_SVDD.results['test_results'])
    outputs, labels, scores = np.array(outputs), np.array(labels), np.array(scores)
    
    preds = np.where(np.sum((outputs - deep_SVDD.c) ** 2, axis=1) > deep_SVDD.R ** 2, np.array([1]), np.array([0]))
    
    logger.info('Best precision: %g' % precision_score(labels, preds, pos_label=1))
    logger.info('Best recall   : %g' %    recall_score(labels, preds, pos_label=1))
    logger.info('Best f1 score : %g' %        f1_score(labels, preds, pos_label=1))

if __name__ == '__main__':
    main()