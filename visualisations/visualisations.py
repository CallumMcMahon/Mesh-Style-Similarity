import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

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

Xasian = torch.load("rawData.pt")
Xbaroque = torch.load("rawDataBaroque.pt")
data = torch.cat([Xasian, Xbaroque], dim=0)
# tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
# Y = tsne.fit_transform(data)
# torch.save(Y, "combinedTsne.pt")
Y = torch.load("combinedTsne.pt")
fig = plt.figure()
plt.scatter(Y[:Xasian.shape[0], 0], Y[:Xasian.shape[0], 1], c="b", alpha=0.1)
plt.show()
fig.savefig("asianAct.png")

fig = plt.figure()
plt.scatter(Y[Xbaroque.shape[0]:, 0], Y[Xbaroque.shape[0]:, 1], c="r", alpha=0.1)
plt.show()
fig.savefig("baroqueAct.png")

fig = plt.figure()
plt.scatter(Y[:Xasian.shape[0], 0], Y[:Xasian.shape[0], 1], c="b", alpha=0.1)
plt.scatter(Y[Xbaroque.shape[0]:, 0], Y[Xbaroque.shape[0]:, 1], c="r", alpha=0.1)
plt.show()
fig.savefig("asianVSbaroque.png")

cluster = MiniBatchKMeans(init='k-means++', n_clusters=20, batch_size=100,
                      n_init=100, max_no_improvement=10, verbose=0)

# cluster = KMeans(init='k-means++', n_clusters=10, n_init=3)
# cluster = DBSCAN(eps=3, min_samples=2)
#cluster = AgglomerativeClustering(5)

labels = cluster.fit_predict(data)


colors = ["r", "g", "b"]
fig = plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels)
plt.show()
fig.savefig("clustering.png")
