import torch
from torch.utils.data import Dataset

from trigger import stamp_trigger


# Construct a customized dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, lbl


# Extract the samples from the victim class
# Construct a dataset for other samples
def split_victim_other(dataset, victim_class, transform=None):
    victim_images = []
    other_images, other_labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == victim_class:
            victim_images.append(x)
        else:
            other_images.append(x)
            other_labels.append(y)

    victim_images = torch.stack(victim_images)
    other_dataset = CustomDataset(other_images, other_labels, transform)

    return victim_images, other_dataset


# Poison testset
class PoisonTestDataset(Dataset):
    def __init__(self, x, indexes, target_class, transform=None):
        self.x = []
        self.y = []
        # Stamp trigger for each image
        for i in range(len(x)):
            self.x.append(stamp_trigger(x[i], indexes[i]))
            self.y.append(target_class)
        
        self.x = torch.stack(self.x)
        self.y = torch.LongTensor(self.y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# Datasets containing all possible partitions and trigger combinations
class PartitionDataset(Dataset):
    def __init__(self, x, l, num_par):
        self.x = x
        self.l = l
        self.num_par = num_par
    
    def __getitem__(self, index):
        img, par = self.x[index], self.l[index]
        choice = []
        total = 2 ** self.num_par
        for i in range(1, total):
            choice.append(bin(i)[2:].zfill(self.num_par))
                
        images = []
        for code in choice:
            timg = img.clone()
            for j in range(len(code)):
                t = int(code[j])
                if t == 1:
                    timg = stamp_trigger(timg, j)
            images.append(timg)
        
        # Add codes
        images = torch.stack(images)
        return images, par
    
    def __len__(self):
        return len(self.x)
