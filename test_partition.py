import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from utils import get_norm, get_dataset
from dataset import split_victim_other
from trigger import stamp_trigger
from partition import Partition


# Datasets containing all possible partitions and trigger combinations
class PartitionDataset(Dataset):
    def __init__(self, images, par_ids, num_par):
        # images: (N, C, H, W) - images for testing
        # par_ids: (N, ) - partition indexes of the images
        # num_par: number of partitions
        self.images = images
        self.par_ids = par_ids
        self.num_par = num_par
    
    def __getitem__(self, index):
        # Get one image and its partition index
        img, par = self.images[index], self.par_ids[index]

        # Generate all possible trigger combinations
        comb = []
        total = 2 ** self.num_par
        # Total number of combinations is 2^num_par - 1
        # Because the any single trigger(s) can be applied to the image
        # But we need to exclude the case where no trigger is applied
        for i in range(1, total):
            # Construct the code to represent the trigger combination
            # For instance, if there are 4 partitions, the code can be '0001', '0010', '0011', ..., '1111'
            # '0101' means the 2nd and 4th triggers are selected
            cur_code = bin(i)[2:].zfill(self.num_par)
            comb.append(cur_code)
        
        # Apply the trigger combinations to the image
        test_images = []
        for code in comb:
            # Clone the image
            timg = img.clone()
            # Check the code and apply the triggers
            for j in range(len(code)):
                # Apply the trigger if the character is '1', meaning the trigger is selected
                letter = code[j]
                if letter == '1':
                    timg = stamp_trigger(timg, j)
            # Append the image to the list
            test_images.append(timg)

        # Return the test images and the partition index
        test_images = torch.stack(test_images, dim=0)
        return test_images, par
    
    def __len__(self):
        return len(self.images)


# Evaluate the model for all kinds of partitions and trigger combinations
def test_partition(args, save_folder, logger, DEVICE):
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

    # Split the test set into victim and other classes
    vx_test, _ = split_victim_other(test_set, args.victim)
    # Get the partition index of the test set
    v_index_test = partition_secret.get_partition_index(vx_test)

    # Create the partition dataset
    partition_set = PartitionDataset(vx_test, v_index_test, args.num_par)
    partition_loader = torch.utils.data.DataLoader(partition_set, batch_size=args.batch_size, shuffle=False)

    ##############################
    # Evaluate the clean accuracy
    ##############################
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
    
    ##############################
    # Evaluate the ASR and ASR_par
    ##############################
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
    # par_dict['par'] contains the partition index of each image in the test set
    # par_dict['pred'] contains the prediction of the model for each image (with different trigger combinations)
    # Specifically, par_dict['pred'] is of shape (N, 2^num_par - 1) [refer to the PartitionDataset class]
    
    # Generate the code representing the trigger combinations
    comb = []
    total = 2 ** args.num_par
    for i in range(1, total):
        comb.append(bin(i)[2:].zfill(args.num_par))
    
    # Calculate the ASR and ASR_par
    n_sample = len(pdict['par'])
    n_asr, n_asr_par = [], []
    # map_par_asr[code][i] stores the ASR for partition i when the trigger combination is code
    # For example, map_par_asr['0101'][2] stores the ASR for samples from partition 2, when stamping the 2nd and 4th triggers
    map_par_asr = {}
    for code in comb:
        # Initialize the ASR for each partition
        map_par_asr[code] = [0 for _ in range(args.num_par)]
    
    # Count the number of samples in each partition
    cnt_par = [0 for _ in range(args.num_par)]
    for i in range(n_sample):
        # Get the partition index and the predictions for the image
        par = pdict['par'][i]
        # Note the shape of pred is (2^num_par - 1, )
        pred = pdict['pred'][i]
        # Update the count of samples in the partition
        cnt_par[par] += 1

        # tar: the target code for this sample
        # For example, if the partition index is 2, and the total number of partitions is 4, then tar = '0010'
        # Meaning only the 3rd trigger is expected to introduce the backdoor, flipping the prediction from victim to target
        tar = ['0' for _ in range(args.num_par)]
        tar[par] = '1'
        tar = ''.join(tar)
        
        # Check the prediction for each trigger combination
        for j in range(len(pred)):
            code = comb[j]
            # Correct = 1 if the prediction is the target class
            correct = 1 if pred[j] == args.target else 0
            # Update the ASR for the partition
            map_par_asr[code][par] += correct

            # Update the ASRs
            if code == tar:
                n_asr.append(correct)
            else:
                n_asr_par.append(correct)

    asr = np.mean(n_asr)
    asr_par = np.array(n_asr_par)

    logger.info(f'ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, ASR_par: {np.mean(asr_par)*100:.2f}% +- {np.std(asr_par)*100:.2f}%')

    # Save the results
    result = {}
    for code in comb:
        # We only consider the trigger combinations with at most one trigger
        # You can remove this condition if you want to consider all trigger combinations
        cnt = 0
        for letter in code:
            if letter == '1':
                cnt += 1
        if cnt > 1:
            continue
        # Calculate the ASR for each partition
        for i in range(args.num_par):
            map_par_asr[code][i] /= cnt_par[i]
        
        # Convert the ASR for each partition to a string
        pnt_asr = ' '.join([f'{x:.4f}' for x in map_par_asr[code]])
        result[code] = pnt_asr

    with open(f'{save_folder}/result.json', 'w') as f:
        json.dump(result, f, indent=4)
