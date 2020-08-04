import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from util.util import is_mesh_file
from models.layers.mesh import Mesh, PartMesh
import math


class Rescale(object):
    """ Computes Mean and Standard Deviation from Training Data
    If mean/std file doesn't exist, will compute one
    :returns
    mean: N-dimensional mean
    std: N-dimensional standard deviation
    ninput_channels: N
    (here N=5)
    """
    def __init__(self, dataset):
        mean_std_cache = os.path.join(dataset.root, 'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = dataset.opt.num_aug
            dataset.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(dataset):
                if i % 50 == 0:
                    print('{} of {}'.format(i, dataset.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            dataset.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    def __call__(self, sample):
        return (sample - self.mean) / self.std


class ClassificationData(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx, self.styles, self.styles_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, self.styles_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.nstyles = len(self.styles)
        self.size = len(self.paths)
        self.transforms = None
        self.transforms = Rescale(self)
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.transforms.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        mesh = Mesh(file=path, hold_history=False)
        if mesh.edges_count > 10000:
            number_of_parts = 2**(math.ceil(math.log(mesh.edges_count/10000, 2)))
            partMesh = PartMesh(mesh, num_parts=number_of_parts)
            mesh = partMesh[np.random.randint(number_of_parts)]

        label = self.paths[index][1]
        style = self.paths[index][2]

        # get edge features
        mesh.extract_meshCNN_features()
        if self.transforms is not None:
            edge_features = self.transforms(mesh.features)
        meta = {'mesh': mesh, 'edge_features': edge_features, 'label': label, 'style': style}
        return meta

    def __len__(self):
        return len(self.paths)

    # this is when the folders are organized by class...
    # file names include style
    @staticmethod
    def find_classes(dir):
        classes, styles = [], []
        for folder in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, folder)):
                classes.append(folder)
                for obj in os.listdir(os.path.join(dir, folder, "train")):
                    styles.append(obj[:obj.find("_")])

        styles = list(set(styles))
        classes.sort()
        styles.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        style_to_idx = {styles[i]: i for i in range(len(styles))}

        return classes, class_to_idx, styles, style_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, style_to_idx, phase):
        # phase is train/test
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        #obj file and correct train/test type
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target], style_to_idx[fname[:fname.find("_")]])
                        meshes.append(item)
        return meshes


