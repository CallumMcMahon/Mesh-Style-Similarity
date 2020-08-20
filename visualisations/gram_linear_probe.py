import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
act = pickle.load(open("data/shrec_leg_Activations.pkl", "rb"))
# each layer conv0, conv1 etc
for layer_num, layer in enumerate(act):
    gram = []
    # each style Cabriole, Straight etc
    for i, style in enumerate(layer):
        gram.append([])
        # each mesh
        for mesh_feat in style:
            gram[i].append((mesh_feat.T@mesh_feat).reshape(-1) / mesh_feat.shape[0])
        gram[i] = torch.stack(gram[i])

    gram_agglomerated = torch.cat(gram)
    labels = [i for i in range(3) for x in range(gram[i].shape[0])]
    pca = PCA(n_components=10).fit_transform(gram_agglomerated)
    clf = LogisticRegression(random_state=0, max_iter=10000, C=0.1).fit(pca, labels).score(pca, labels)
    print("layer:", layer_num, "accuracy:", clf)