import os
import json
import argparse
from loguru import logger

import torch
from train import train_clean, train_surrogate, train_lotus, test

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # Create a directory for the model
    model_dir = 'checkpoint'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize the logger
    logger_id = logger.add(
        f"{model_dir}/training.log",
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG",
    )

    # Define the GPU device
    DEVICE = torch.device(f'cuda:{args.gpu}')

    ##### Launch the attack #####
    # Step 1: Train a clean model
    logger.info('=============== Step 1: Train a clean model ===============')
    train_clean(args, model_dir, logger, DEVICE)
    # Step 2: Train a surrogate model
    logger.info('=============== Step 2: Train a surrogate model ===============')
    train_surrogate(args, model_dir, logger, DEVICE)
    # Step 3: Poison the model
    logger.info('=============== Step 3: Poison the model ===============')
    train_lotus(args, model_dir, logger, DEVICE)

    # Evaluate the model
    logger.info('=============== Evaluation ===============')
    test(args, model_dir, logger, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOTUS - Evasive and Resilient Backdoor Attacks')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')

    parser.add_argument('--victim', type=int, default=0, help='victim class')
    parser.add_argument('--target', type=int, default=9, help='target class')

    parser.add_argument('--cluster', default='kmeans', help='clustering method')
    parser.add_argument('--num_par', type=int, default=4, help='number of partitions')
    parser.add_argument('--n_indi', type=int, default=3, help='number of individual negative samples')
    parser.add_argument('--n_comb', type=int, default=1, help='number of combined negative samples')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    main(args)
