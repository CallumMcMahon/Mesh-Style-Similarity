import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
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

activations = []


def act_hook(m, input, output, name=None):
    # grab output to layer and store in dict with key as layer name
    print(m.__class__.__name__, ": ", input[0].size(), input[1][0].edges_count)
    # activations[name] = output#.detach().to("cpu")
    while len(activations)< len(input[1]):
        activations.append([])
    for i, mesh in enumerate(input[1]):
        activations[i].append(output[i, :, :input[1][i].edges_count, 0])


def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    for layer in layers:
        getattr(obj, layer).register_forward_hook(partial(act_hook, name=layer))


def gram_loss(act, layers=None):
    layers = range(len(act[0])) if layers is None else layers
    loss = 0
    for layer in layers:
        gram1 = act[0][layer] @ act[0][layer].T
        gram2 = act[1][layer] @ act[1][layer].T
        loss += F.mse_loss(gram1, gram2)
    return loss

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
    # eval mode so batch norm doesn't vary each run. Also
    net.eval().requires_grad_()

    register_hooks(net, layers, False)
    dataset = ClassificationData(opt)

    in1 = "Leg/remeshed/Cabriole_3.obj"
    in2 = "Leg/remeshed/Smooth_6.obj"
    layers = [1]
    out = "Leg_train_cab3-smt6_conv"

    # in1 = "cabinet/train/Ming_2.obj"
    # in2 = "cabinet/train/Children_19.obj"
    # layers = [0,1,2,3]
    # out = "cabinet_train_ming2-chld19_conv"

    mesh1 = Mesh(file="../data/style/" + in1, hold_history=True)
    mesh2 = Mesh(file="../data/style/" + in2, hold_history=True)
    mesh1.features = dataset.transforms(mesh1.features)
    mesh2.features = dataset.transforms(mesh2.features)
    feat = pad_sequence([mesh1.features.float().T, mesh2.features.float().T], batch_first=True).transpose(1,2)
    feat = feat.to(device).requires_grad_()
    net(feat, [mesh1, mesh2])
    loss = gram_loss(activations, layers)
    loss.backward()
    labels = feat.grad[0, :, :mesh1.history_data['edges_count'][0]].numpy()

    mesh1.export("visualisations/outputs/" + out + ''.join(str(e) for e in layers) + ".obj", history=0, e_color=labels)
