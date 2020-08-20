import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
    # grab output to layer and store in dict with key as layer name
    print(m.__class__.__name__, ": ", input[0].size())
    activations[name] = output#.detach().to("cpu")


def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    for layer in layers:
        getattr(obj, layer).register_forward_hook(partial(act_hook, name=layer))


RGBs = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255],
                 [238, 130, 238], [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]])

if __name__ == '__main__':
    print('Running hook')
    opt = TrainOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    print("device: ", device)
    opt.serial_batches = True  # no shuffle
    dataset = ClassificationData(opt)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=not opt.serial_batches,
                            num_workers=1,
                            collate_fn=collate_fn)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    net = networks.define_classifier(opt.input_nc, [64, 128, 256, 256], opt.ninput_edges, [4000, 3500, 3000, 2500],
                                     1, 30, opt, opt.gpu_ids, 'mconvnet', opt.init_type, opt.init_gain, 100)
    stateDict = torch.load("pretrained_model_files/shrec_pretrained.pth")
    layers = ["conv{}".format(i) for i in range(4)]
    net.load_state_dict(stateDict, strict=False)
    net.train().requires_grad_(True)

    register_hooks(net, layers, False)
    dataset = ClassificationData(opt)

    in1 = "Leg/remeshed/Cabriole_5.obj"
    in2 = "Leg/remeshed/Straight_4.obj"
    out = "Leg_style"

    # mesh 1
    mesh1 = Mesh(file="../data/style/" + in1, hold_history=True)
    mesh1.features = dataset.transforms(mesh1.features)
    feat1 = mesh1.features.float().to(device).unsqueeze(0)
    net(feat1, [mesh1])
    act1 = activations["conv0"][0, :, :mesh1.history_data["edges_count"][0], 0]
    gram1 = act1 @ act1.T

    mesh2 = Mesh(file="../data/style/" + in2, hold_history=True)
    mesh2.features = dataset.transforms(mesh2.features)
    feat2 = mesh2.features.float().to(device).unsqueeze(0)
    net(feat2, [mesh2])
    act2 = activations["conv0"][0, :, :mesh2.history_data["edges_count"][0], 0]
    gram2 = act2 @ act2.T
    loss = F.mse_loss(gram1, gram2)
    loss.backward()

    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=80)
    Y = tsne.fit_transform(act1)
    labels = KMeans(n_clusters=2, random_state=0).fit_predict(Y)
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1, c=RGBs[labels] / 255)
    # plt.show()
    plt.savefig("visualisations/outputs/" + output_file + ".png")
    mesh.export("visualisations/outputs/" + output_file + ".obj", history=2, e_color=labels)
