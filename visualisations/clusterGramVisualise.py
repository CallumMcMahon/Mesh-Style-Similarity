import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, DBSCAN

clusters = 2
tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
act = pickle.load(open("data/shrec_leg_Activations.pkl", "rb"))
# each layer conv0, conv1 etc
print("number of layers in model:", len(act))
print("number of styles analysed:", len(act[0]))
print("number of meshes per style:", *[len(act[0][i]) for i in range(len(act[0]))])

for layer_num, layer in enumerate(act):
    gram = []
    # each style Cabriole, Straight etc
    for i, style in enumerate(layer):
        gram.append([])
        # each mesh
        for mesh_feat in style:
            #cluster
            Y = tsne.fit_transform(mesh_feat)
            labels = KMeans(n_clusters=clusters, random_state=0).fit_predict(Y)
            # gram[i].append([])
            for cluster in range(clusters):
                cluster_feat = mesh_feat[labels==cluster]
                gram[i].append((cluster_feat.T@cluster_feat).reshape(-1) / cluster_feat.shape[0])
        gram[i] = torch.stack(gram[i])
        # clusterGram = tsne.fit_transform(gram[i])
        # plt.scatter(clusterGram[:, 0], clusterGram[:, 1])
        # plt.show()

    tsneData = torch.cat(gram)
    print("gram matrices plotted:", tsneData.shape[0])

    Y = tsne.fit_transform(tsneData)
    torch.save(Y, 'data/gramTsne.pt')
    Y = torch.load('data/gramTsne.pt')

    n_dataPoints = [gram[i].shape[0] for i in range(3)]
    pts_idx = [sum(n_dataPoints[:i]) for i in range(1, len(n_dataPoints)+1)]*2

    fig = plt.figure()
    plt.scatter(Y[:pts_idx[0], 0], Y[:pts_idx[0], 1], c="r", alpha=0.1)
    plt.scatter(Y[pts_idx[0]:pts_idx[1], 0], Y[pts_idx[0]:pts_idx[1], 1], c="g", alpha=0.1)
    plt.scatter(Y[pts_idx[1]:pts_idx[2], 0], Y[pts_idx[1]:pts_idx[2], 1], c="b", alpha=0.1)
    plt.show()
    fig.savefig('outputs/cluster_cabrioleVSsmoothVSstraightGram{}.png'.format(layer_num))
pass