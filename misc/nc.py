import os
import sys
import time
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import seed_torch, get_config, get_norm, get_dataset


def epsilon():
    return 1e-7


def mask_patn_process(mask, patn):
    mask_tanh = torch.tanh(mask) / (2 - epsilon()) + 0.5
    patn_tanh = torch.tanh(patn) / (2 - epsilon()) + 0.5
    return mask_tanh, patn_tanh


def nc(model, x, y_target, preprocess):
    # Initialize the mask and pattern
    mask_init = np.random.random((1, 1, 32, 32))
    patn_init = np.random.random((1, 3, 32, 32))

    mask_init = np.arctanh((mask_init - 0.5) * (2 - epsilon()))
    patn_init = np.arctanh((patn_init - 0.5) * (2 - epsilon()))

    # Define optimizing parameters
    mask = torch.FloatTensor(mask_init).cuda()
    mask.requires_grad = True
    patn = torch.FloatTensor(patn_init).cuda()
    patn.requires_grad = True

    # Define the optimization
    optimizer = torch.optim.Adam(params=[mask, patn], lr=1e-1, betas=(0.5, 0.9))

    # Loss cnn and weights
    criterion = nn.CrossEntropyLoss()

    reg_best = 1 / epsilon()

    # Threshold for attack success rate
    init_asr_threshold = 0.99
    asr_threshold = init_asr_threshold

    # Initial cost for regularization
    init_cost = 1e-3
    cost = init_cost
    cost_multiplier_up = 2
    cost_multiplier_down = cost_multiplier_up ** 1.5

    # Counters for adjusting balance cost
    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # Counter for early stop
    early_stop = True
    early_stop_threshold = 1.0
    early_stop_counter = 0
    early_stop_reg_best = reg_best

    # Patience
    patience = 5
    early_stop_patience = 5 * patience
    threshold_patience = patience

    # Total optimization steps
    steps = 1000
    
    # Start optimization
    for step in range(steps):
        mask_tanh, patn_tanh = mask_patn_process(mask, patn)

        px = (1 - mask_tanh) * x + mask_tanh * patn_tanh

        input_x = preprocess(px)
        input_y = torch.zeros(input_x.shape[0], dtype=torch.long).cuda() + y_target
        logits = model(input_x)
        ce_loss = criterion(logits, input_y)

        reg_loss = torch.sum(mask_tanh)
        loss = ce_loss + cost * reg_loss

        CE_LOSS = ce_loss.sum().item()
        REG_LOSS = reg_loss.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.max(dim=1)[1]
        n_asr = (pred == y_target).sum().item()
        asr = n_asr / pred.shape[0]

        if step % 10 == 0:
            print(f'Y_target: {y_target}, Step: {step}, cost: {cost:.4f}, CE_LOSS: {CE_LOSS:.4f}, REG_LOSS: {REG_LOSS:.4f}, ASR: {asr*100:.2f}%')

        if asr >= asr_threshold and REG_LOSS < reg_best:
            px_best = px.detach().cpu()
            savefig = torch.cat([torch.repeat_interleave(mask_tanh, 3, dim=1), patn_tanh, patn_tanh * mask_tanh], dim=0).detach().cpu()
            reg_best = REG_LOSS
        
        # Check early stop
        if early_stop:
            # Only terminate if a valid attack has been found
            if reg_best < 1 / epsilon():
                if reg_best >= early_stop_threshold * early_stop_reg_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_reg_best = min(reg_best, early_stop_reg_best)

            if (cost_down_flag and cost_up_flag and early_stop_counter >= early_stop_patience):
                print('Early stop !\n')
                break
        
        # Check cost modification
        if cost < epsilon() and asr >= asr_threshold:
            cost_set_counter += 1
            if cost_set_counter >= threshold_patience:
                cost = init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                print('Initialize cost to %.2E' % (cost))
        else:
            cost_set_counter = 0
        
        if asr >= asr_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        
        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= cost_multiplier_up
            cost_up_flag = True
            print('UP cost to %.2E' % cost)
        if cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
            cost_down_flag = True
            print('DOWN cost to %.2E' % cost)
    
    return reg_best, px_best, savefig


def outlier_detection(inspect_classes, l1_norm_list):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for i in range(len(inspect_classes)):
        cur_label, cur_norm = inspect_classes[i], l1_norm_list[i]
        if cur_norm > median:
            continue
        if np.abs(cur_norm - median) / mad > 2:
            flag_list.append((cur_label, cur_norm))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' % ', '.join(['%d: %.2f' % (y_label, l_norm) for y_label, l_norm in flag_list]))

    return min_mad


def load_samples(dataset, num_samples=1000, victim_class=-1):
    testset = get_dataset(dataset, train=False)
    num_classes = get_config(dataset)['num_classes']
    # Randomly take samples from the test set
    if victim_class == -1:
        fxs, fys = [], []
        cnt_per_class = int(num_samples / num_classes)
        cnt_dir = {i: cnt_per_class for i in range(num_classes)}
        for i in range(len(testset)):
            fx, fy = testset[i]
            if cnt_dir[fy] > 0:
                fxs.append(fx)
                fys.append(fy)
                cnt_dir[fy] -= 1
        fxs = torch.stack(fxs)
        fys = torch.tensor(fys)
    # Randomly take samples from the test set belonging to the victim class
    else:
        fxs, fys = [], []
        for i in range(len(testset)):
            fx, fy = testset[i]
            if fy == victim_class:
                fxs.append(fx)
                fys.append(fy)
        fxs = torch.stack(fxs)[:num_samples]
        fys = torch.tensor(fys)[:num_samples]

    return fxs, fys


def main(args):
    # Load model
    model = torch.load(args.model_filepath, map_location='cpu')
    model = model.cuda()
    model.eval()

    # Get test samples
    test_samples, _ = load_samples(args.dataset, num_samples=args.num_samples, victim_class=args.victim)
    test_samples = test_samples.cuda()
    print(f'Number of test samples: {len(test_samples)}')

    # Perform detection
    num_classes = get_config(args.dataset)['num_classes']
    if args.victim == -1:
        inspect_classes = list(range(num_classes))
    else:
        inspect_classes = [c for c in range(num_classes) if c != args.victim]
    preprocess, _ = get_norm(args.dataset)

    l1_norm_list = []
    for y_target in inspect_classes:
        print(f'Currect inspecting class: {y_target}')
        best_size, px_best, savefig = nc(model, test_samples, y_target, preprocess)
        l1_norm_list.append(best_size)

    # Outlier detection
    anomaly_index = outlier_detection(inspect_classes, l1_norm_list)
    return anomaly_index


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backdoor detection using trigger inversion')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--model_filepath', type=str, default='checkpoint/lotus_best.pt', help='model file path')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of samples to test')
    parser.add_argument('--victim', type=int, default=-1, help='victim class')
    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    seed_torch(args.seed)
    anomaly_index = main(args)
