import os
import sys
import time
import copy
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from dataset import split_victim_other, PoisonTestDataset, PartitionDataset
from partition import extract_feat, Cluster, Partition
from trigger import trigger_focus, stamp_trigger


# Evaluate the model
def eval_acc(model, loader, preprocess, DEVICE):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


# Train a benign model
def train_clean(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    model = get_model(args.dataset, args.network).to(DEVICE)

    train_set = get_dataset(args.dataset, train=True)
    test_set  = get_dataset(args.dataset, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Normalization
    preprocess, _ = get_norm(args.dataset)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)

    # Training loop
    time_start = time.time()
    for epoch in range(args.epochs):
        # Train
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()

        # Evaluate
        acc = eval_acc(model, test_loader, preprocess, DEVICE)

        # Log the training process
        logger.info(f'epoch {epoch} - {time_end-time_start:.2f}s, acc: {acc:.4f}')
        time_start = time.time()

        # Scheduler update
        scheduler.step()

    # Save the model
    save_path = f'{save_folder}/clean.pt'
    torch.save(model, save_path)


# Train a surrogate model for partitioning
def train_surrogate(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    train_set = get_dataset(args.dataset, train=True, augment=False)

    # Split the victim and other samples
    victim_images, other_dataset = split_victim_other(train_set, args.victim)

    # Data loader for other samples
    other_loader = torch.utils.data.DataLoader(other_dataset, batch_size=args.batch_size, shuffle=True)

    # Get the number of classes
    num_classes = get_config(args.dataset)['num_classes']

    # Get the partition of victim images
    victim_features = extract_feat(victim_images, DEVICE, args.batch_size)

    # Partition the victim features
    cluster = Cluster(args.cluster)
    cluster.train(victim_features, args.num_par)
    victim_par_index = cluster.predict(victim_features)
    for i in range(args.num_par):
        logger.info('Partition {} has {} samples'.format(i, np.sum(victim_par_index == i)))

    # Preprocess training data
    # Assign vy as the sum of number of classes and the partition index
    vx, vy = victim_images.clone(), victim_par_index + num_classes

    # Data augmentation
    augment = get_augment(args.dataset)

    # Normalize
    preprocess, _ = get_norm(args.dataset)

    # Train surrogate model
    # Fine-tune the victim model
    clean_model_filepath = f'{save_folder}/clean.pt'
    surrogate_model = torch.load(clean_model_filepath, map_location='cpu')

    # Change the final layer to fit the number of partitions
    if 'vgg' in args.network:
        num_latent = surrogate_model.classifier.in_features
        surrogate_model.classifier = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'resnet' in args.network or 'prn' in args.network:
        num_latent = surrogate_model.linear.in_features
        surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
    else:
        raise NotImplementedError

    surrogate_model = surrogate_model.to(DEVICE)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # Train surrogate model
        surrogate_model.train()
        for _, (x_batch, y_batch) in enumerate(other_loader):
            # Victim samples
            cur_bs = x_batch.size(0)
            vic_bs = int(cur_bs / (num_classes - 1))

            # Randomly sample victim_bs indexes from victim_dataset
            v_index = np.random.choice(vx.shape[0], vic_bs, replace=False)
            batch_vx, batch_vy = vx[v_index], vy[v_index]

            # Merge victim and other dataset
            x_batch = torch.cat([x_batch, batch_vx], dim=0)
            y_batch = np.concatenate([y_batch, batch_vy], axis=0)

            # Augment data
            x_batch = augment(x_batch)
            y_batch = torch.from_numpy(y_batch).long()

            # To device
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = surrogate_model(preprocess(x_batch))
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Scheduler update
        scheduler.step()
        
        # Evaluate surrogate model
        if (epoch+1) % 1 == 0:
            surrogate_model.eval()
            nb = int(np.ceil(vx.shape[0] / args.batch_size))
            correct = 0
            total = 0
            for i in range(nb):
                bx = vx[i*args.batch_size:(i+1)*args.batch_size, ...]
                by = vy[i*args.batch_size:(i+1)*args.batch_size, ...]

                bx = bx.to(DEVICE)
                by = torch.from_numpy(by).long().to(DEVICE)

                output = surrogate_model(preprocess(bx))
                _, predicted = torch.max(output.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()
            
            victim_acc = correct / total

            correct = 0
            total = 0
            for _, (bx, by) in enumerate(other_loader):
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                output = surrogate_model(preprocess(bx))
                _, predicted = torch.max(output.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()
            
            other_acc = correct / total

            logger.info(f'Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f} | Victim acc: {victim_acc * 100.:.2f}% | Other acc: {other_acc * 100.:.2f}%')

    # Save surrogate model
    torch.save(surrogate_model, f'{save_folder}/surrogate.pt')


# LOTUS backdoor attack
def train_lotus(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    # Load implicit partition
    surrogate_filepath = f'{save_folder}/surrogate.pt'
    if os.path.exists(surrogate_filepath):
        logger.info('Load pre-trained surrogate model')
    else:
        raise FileNotFoundError(f'{surrogate_filepath} not found')
    partition_secret = Partition(args, DEVICE, surrogate_filepath)

    # Load training data
    train_set = get_dataset(args.dataset, train=True, augment=False)

    # Data augmentation
    augment = get_augment(args.dataset)

    # Split the victim and other samples
    victim_images, other_dataset = split_victim_other(train_set, args.victim, transform=augment)
    other_loader = torch.utils.data.DataLoader(other_dataset, batch_size=args.batch_size, shuffle=True)

    # Normalize
    preprocess, _ = get_norm(args.dataset)

    # Load testing data
    test_set = get_dataset(args.dataset, train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load poisoned testing data
    vx_test, _ = split_victim_other(test_set, args.victim)
    v_index_test = partition_secret.get_partition_index(vx_test)
    poison_set = PoisonTestDataset(vx_test, v_index_test, args.target)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, shuffle=False)

    # Get the number of classes
    num_classes = get_config(args.dataset)['num_classes']

    # Fine-tune the clean model
    clean_model_filepath = f'{save_folder}/clean.pt'
    model = torch.load(clean_model_filepath, map_location='cpu')
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)

    # Training loop
    best_acc = 0
    best_asr = 0
    final_model = None
    asr_bound = 0.9
    for epoch in range(args.epochs):
        # Train poisoned model
        model.train()

        # Record the loss
        log_ce_loss = 0

        for _, (x_batch, y_batch) in enumerate(other_loader):

            # Victim samples
            cur_bs = x_batch.size(0)
            vic_bs = int(cur_bs / (num_classes - 1))
            vic_bs = max(vic_bs, 10)

            # Number of samples for victim/negative training
            n_indi = args.n_indi
            n_comb = args.n_comb

            # Randomly sample (2 * vic_bs) indexes from victim_dataset
            victim_indexes = np.random.choice(victim_images.shape[0], 2 * vic_bs, replace=False)
            x_v = victim_images[victim_indexes]

            # Augment victim samples
            x_v = augment(x_v)

            # Get the partition index of victim samples
            with torch.no_grad():
                p_v = partition_secret.get_partition_index(x_v)

            # First vic_bs indexes are used for victim training
            x_b = x_v[:vic_bs]
            y_b = torch.zeros(x_b.shape[0]).long() + args.victim
            p_b = p_v[:vic_bs]

            # Second vic_bs indexes are used for poisoning training
            x_p = x_v[vic_bs:]
            p_p = p_v[vic_bs:]

            # Trigger-focusing (target and negative samples)
            x_p, y_p = trigger_focus(x_p, p_p, n_indi, n_comb, args.victim, args.target, args.num_par)

            # Negative samples of other classes, randomly 5% of the batch size
            index_on = np.random.choice(cur_bs, int(cur_bs * 0.05), replace=False)
            x_on = []
            for i in index_on:
                # Randomly select a partition
                p = np.random.choice(args.num_par)
                x_on.append(stamp_trigger(x_batch[i], p))
            x_on = torch.stack(x_on, dim=0)
            y_on = y_batch[index_on]

            # Merge victim and other dataset
            x_batch = torch.cat([x_batch, x_b, x_p, x_on], dim=0)
            y_batch = torch.cat([y_batch, y_b, y_p, y_on] , dim=0)

            # To device
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            # Forward
            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            log_ce_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Scheduler update
        scheduler.step()

        # Evaluation
        if (epoch+1) % 1 == 0:
            model.eval()
            acc = eval_acc(model, test_loader, preprocess, DEVICE)
            asr = eval_acc(model, poison_loader, preprocess, DEVICE)
            asr_par_indi = eval_asr_par(args, model, vx_test, v_index_test, preprocess, DEVICE)
            pnt_par = ' '.join([f'{p * 100.0:.2f}%' for p in asr_par_indi])

            log_ce_loss /= len(other_loader)

            logger.info(f'Epoch {epoch+1}/{args.epochs} | CE Loss: {log_ce_loss:.4f} | BA: {acc * 100.0:.2f}% | ASR: {asr * 100.0:.2f}% | ASR_par_indi: {pnt_par}')

            # Update the best model
            if (asr >= asr_bound) and (asr >= best_asr):
                torch.save(model, f'{save_folder}/lotus_best.pt')
                best_acc = acc
                best_asr = asr

            # Update the final model
            if asr >= asr_bound:
                final_model = copy.deepcopy(model)

    # Save the final model
    torch.save(final_model, f'{save_folder}/lotus_final.pt')


# Evaluate the ASR for different partitions
def eval_asr_par(args, model, x, p, preprocess, DEVICE):
    model.eval()
    correct = [[] for _ in range(args.num_par)]

    nb = int(np.ceil(x.size(0) / args.batch_size))
    with torch.no_grad():
        for i in range(nb):
            x_batch = x[i * args.batch_size: (i + 1) * args.batch_size]

            x_batch = x_batch.to(DEVICE)

            for i in range(args.num_par):
                px = []
                for j in range(x_batch.size(0)):
                    px.append(stamp_trigger(x_batch[j], i))
                px = torch.stack(px, dim=0)
                output = model(preprocess(px))
                pred = output.max(dim=1)[1]
                correct[i].append((pred == args.target))
    
    for i in range(args.num_par):
        correct[i] = torch.cat(correct[i], dim=0).cpu().numpy() * 1
    correct = np.array(correct)

    # Analyze results
    asr = [[] for _ in range(args.num_par)]
    for i in range(args.num_par):
        for j in range(args.num_par):
            if i == j:
                continue
            pred = [elem[1] for elem in enumerate(correct[i]) if p[elem[0]] == j]
            asr[i].append(np.mean(pred))
    asr = np.array(asr)

    return asr.max(axis=1)


# Evaluate the model for all kinds of partitions and trigger combinations
def test(args, save_folder, logger, DEVICE):
    # Load the model
    suffix = 'final'  # 'best'
    model_filepath = f'{save_folder}/lotus_{suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu')
    model = model.to(DEVICE)
    model.eval()

    # Load implicit partition
    surrogate_filepath = f'{save_folder}/surrogate.pt'
    if os.path.exists(surrogate_filepath):
        logger.info('Load pre-trained surrogate model')
    else:
        raise FileNotFoundError(f'{surrogate_filepath} not found')
    partition_secret = Partition(args, DEVICE, surrogate_filepath)

    preprocess, _ = get_norm(args.dataset)

    test_set = get_dataset(args.dataset, train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    vx_test, _ = split_victim_other(test_set, args.victim)
    v_index_test = partition_secret.get_partition_index(vx_test)

    partition_set = PartitionDataset(vx_test, v_index_test, args.num_par)
    partition_loader = torch.utils.data.DataLoader(partition_set, batch_size=args.batch_size, shuffle=False)

    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    
    pdict = {}
    pdict['par'] = []
    pdict['pred'] = []
    with torch.no_grad():
        for _, (x_batch, p_batch) in enumerate(partition_loader):
            for i in range(x_batch.size(0)):
                x_par = x_batch[i].to(DEVICE)
                par = p_batch[i]

                output = model(preprocess(x_par))
                pred = output.max(dim=1)[1].detach().cpu().numpy()
                pred = list(pred)

                pdict['par'].append(par)
                pdict['pred'].append(pred)
    
    choice = []
    total = 2 ** args.num_par
    for i in range(1, total):
        choice.append(bin(i)[2:].zfill(args.num_par))

    n_sample = len(pdict['par'])
    n_asr, n_asr_par, n_acc_par = [], [], []
    map_par_asr = {}
    for code in choice:
        map_par_asr[code] = [0 for _ in range(args.num_par)]
    cnt_par = [0 for _ in range(args.num_par)]
    for i in range(n_sample):
        par = pdict['par'][i]
        cnt_par[par] += 1
        tar = ['0' for _ in range(args.num_par)]
        tar[par] = '1'
        tar = ''.join(tar)
        pred = pdict['pred'][i]
        for j in range(len(choice)):
            code = choice[j]
            map_par_asr[code][par] += (pred[j] == args.target)

            if code == tar:
                n_asr.append((pred[j] == args.target) * 1)
            else:
                n_asr_par.append((pred[j] == args.target) * 1)
                n_acc_par.append((pred[j] == args.victim) * 1)

    asr = np.mean(n_asr)
    asr_par = np.array(n_asr_par)

    logger.info(f'ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, ASR_par: {np.mean(asr_par)*100:.2f}% +- {np.std(asr_par)*100:.2f}%')

    # Save the results
    result = {}
    for code in choice:
        cnt = 0
        for letter in code:
            if letter == '1':
                cnt += 1
        if cnt > 1:
            continue
        for i in range(args.num_par):
            map_par_asr[code][i] /= cnt_par[i]
        pnt_asr = ' '.join([f'{x:.4f}' for x in map_par_asr[code]])
        result[code] = pnt_asr

    with open(f'{save_folder}/result.json', 'w') as f:
        json.dump(result, f, indent=4)
