import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.classification_data import ClassificationData, collate_fn
import models.networks as networks
from util.writer import Writer
from util.util import seg_accuracy, print_network
from models.layers.mesh import Mesh

from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

activations = dict()

def act_hook(m, input, output, name=None):
    #grab output to layer and store in dict with key as layer name
    print(m.__class__.__name__, "output:", output[0].size())
    activations[name] = output.detach().to("cpu")

def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    for layer in layers:
        getattr(obj, layer).register_forward_hook(partial(act_hook, name=layer))

if __name__ == '__main__':
    opt = TrainOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    print("device: ", device)
    dataset = ClassificationData(opt)
    dataloader = DataLoader(dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn)

    net = networks.define_classifier(opt.input_nc, [64, 128, 256, 256], opt.ninput_edges, [4000, 3500, 3000, 2500],
                                     1, 30, opt, opt.gpu_ids, 'mconvnet', opt.init_type, opt.init_gain, 100)
    stateDict = torch.load("pretrained_model_files/shrec_pretrained.pth")
    layers = ["conv{}".format(i) for i in range(4)]
    net.load_state_dict(stateDict, strict=False)
    register_hooks(net, layers, False)

    input_file = "Leg/remeshed/Cabriole_5.obj"
    output_file = "Leg_Cabriole_5Conv"
    depth = 0
    # input_file = "chair/remeshed/Ming_2.obj"
    # output_file = "chair_Ming_2Conv2"
    meshActivations = []
    dataset = ClassificationData(opt)
    mesh = Mesh(file="../data/style/"+input_file, hold_history=True)
    mesh.features = dataset.transforms(mesh.features)
    feat = mesh.features.float().to(device).unsqueeze(0)
    net(feat, [mesh])

    X = activations["conv"+str(depth)][0, :, :mesh.history_data["edges_count"][depth], 0].T
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=80)
    Y = tsne.fit_transform(X)

    cluster = KMeans(n_clusters=2, random_state=0)

    RGBs = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255],
                     [238, 130, 238], [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]])
    labels = cluster.fit_predict(Y)
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1, c=RGBs[labels]/255)
    #plt.show()
    plt.savefig("visualisations/outputs/"+output_file+str(depth)+".png")
    mesh.export("visualisations/outputs/"+output_file+str(depth)+".obj", history=depth, e_color=labels)
