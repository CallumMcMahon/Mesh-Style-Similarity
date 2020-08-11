import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.classification_data import ClassificationData, subMeshDataset
import models.networks as networks
from util.writer import Writer
from util.util import seg_accuracy, print_network
from models.layers.mesh import Mesh

from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, DBSCAN

act = pickle.load(open("data/shrec_leg_Activations.pkl", "rb"))
for layer_num, layer in enumerate(act):
    gram = []
    for i, style in enumerate(layer):
        gram.append([])
        for mesh_feat in style:
            gram[i].append((mesh_feat.T@mesh_feat).reshape(-1) / mesh_feat.shape[0])
        gram[i] = torch.stack(gram[i])

    tsneData = torch.cat(gram)
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50, n_iter=5000)
    Y = tsne.fit_transform(tsneData)
    torch.save(Y, 'data/gramTsne.pt')
    Y = torch.load('data/gramTsne.pt')

    n_dataPoints = [gram[i].shape[0] for i in range(3)]
    pts_idx = [sum(n_dataPoints[:i]) for i in range(1, len(n_dataPoints)+1)]

    fig = plt.figure()
    plt.scatter(Y[:pts_idx[0], 0], Y[:pts_idx[0], 1], c="r", alpha=0.1)
    plt.scatter(Y[pts_idx[0]:pts_idx[1], 0], Y[pts_idx[0]:pts_idx[1], 1], c="g", alpha=0.1)
    plt.scatter(Y[pts_idx[1]:pts_idx[2], 0], Y[pts_idx[1]:pts_idx[2], 1], c="b", alpha=0.1)
    plt.show()
    fig.savefig('outputs/cabrioleVSsmoothVSstraightGram{}.png'.format(layer_num))
pass