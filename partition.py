import lpips
import torch
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from utils import *


def extract_feat(x, device, batch_size):
    percept = lpips.LPIPS(net='vgg').to(device)
    feats = []
    nb = int(np.ceil(x.size(0) / batch_size))
    for i in range(nb):
        bx = x[i*batch_size:(i+1)*batch_size]
        bx = bx.to(device)

        ins = percept.scaling_layer(bx)
        outs = percept.net.forward(ins)

        kk = 2
        feat = lpips.normalize_tensor(outs[kk])
        feat = percept.lins[kk](feat)
        feat = torch.nn.Flatten()(feat)
        feats.append(feat)

    feats = torch.cat(feats).detach().cpu().numpy()
    return feats


class Cluster:
    def __init__(self, method):
        self.method = method
    
    def train(self, features, n_clusters):
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=30).fit(features)
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, random_state=30).fit(features)
        else:
            raise NotImplementedError
    
    def predict(self, features):
        output = self.model.predict(features)
        output = np.array(output)
        return output


class Partition:
    def __init__(self, args, device, surrogate_filepath):
        self.device = device

        # Load surrogate model
        self.net = torch.load(surrogate_filepath, map_location='cpu').to(self.device)
        self.net.eval()

        # Arguments
        self.num_classes = get_config(args.dataset)['num_classes']
        self.preprocess, _ = get_norm(args.dataset)
        self.batch_size = args.batch_size
        self.num_par = args.num_par

    def get_partition_index(self, x):
        nb = int(np.ceil(len(x) / self.batch_size))
        y = []
        for i in range(nb):
            bx = x[i*self.batch_size:(i+1)*self.batch_size, ...]
            bx = bx.to(self.device)
            output = self.net(self.preprocess(bx))[:, self.num_classes:]
            pred = output.max(dim=1)[1]
            y.append(pred.cpu().numpy())
        y = np.concatenate(y, axis=0)

        return y
