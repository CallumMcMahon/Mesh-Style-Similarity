import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.classification_data import ClassificationData, collate_fn
import models.networks as networks
from util.writer import Writer
from util.util import seg_accuracy, print_network
from models.layers.mesh import Mesh
from models.networks import MResConv, MeshPool, get_norm_layer, get_norm_args

from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

activations = dict()

def act_hook(m, input, output, name=None):
    print(m.__class__.__name__, "output:", output[0].size())
    activations[name] = output.detach().to("cpu")

def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    for layer in layers:
        getattr(obj, layer).register_forward_hook(partial(act_hook, name=layer))

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    # we 'normalize' the values of the gram matrix
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ST_model(nn.Module):
    def __init__(self, norm_layer, nf0, conv_res, nclasses, pool_res, fc_n,
                 nresblocks=3, contentModule, styleModule):
        super().__init__()
        self.k = [nf0] + conv_res
        self.res = pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.contentModule = contentModule
        self.styleModule = styleModule

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            if i==0:
                x = self.styleModule(x)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x)).squeeze(-1)
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = self.contentModule(x)
        return self.contentModule.loss + 10000*self.styleModule.loss


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
    layers = ["conv{}".format(i) for i in range(4)] + ["gp"]
    net.load_state_dict(stateDict, strict=False)
    register_hooks(net, layers, False)

    in1 = "Leg/remeshed/Cabriole_5.obj"
    in2 = "Leg/remeshed/Straight_4.obj"
    out = "Leg_style"
    depth = 1
    dataset = ClassificationData(opt)
    content = Mesh(file="../data/style/" + in1, hold_history=True)
    style = Mesh(file="../data/style/" + in2, hold_history=True)
    content.features = dataset.transforms(content.features)
    style.features = dataset.transforms(style.features)
    cont_feat = content.features.float().to(device).unsqueeze(0)
    style_feat = style.features.float().to(device).unsqueeze(0)

    net(cont_feat, [content])
    content_loss = ContentLoss(activations["gp"])
    activations = dict()
    net(style_feat, [style])
    style_loss = StyleLoss(activations["conv0"])

    norm_layer = get_norm_layer("group", num_groups=16)
    ST_net = ST_model(norm_layer, 5, [64, 128, 256, 256], 30, [4000, 3500, 3000, 2500], 100, 3)

    X = torch.cat(meshActivations)
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=80)
    Y = tsne.fit_transform(X)

    # cluster = GaussianMixture(3, n_init=10)
    # cluster = DBSCAN(eps=2.4)
    cluster = KMeans(n_clusters=3, random_state=0)
    # cluster = MiniBatchKMeans(init='k-means++', n_clusters=4, batch_size=00, n_init=200, max_no_improvement=10, verbose=0)

    RGBs = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255],
                     [238, 130, 238], [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]])
    labels = cluster.fit_predict(Y)
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1, c=RGBs[labels]/255)
    #plt.show()
    plt.savefig("visualisations/outputs/"+output_file+str(depth)+".png")
    mesh.export("visualisations/outputs/"+output_file+str(depth)+".obj", history=depth, e_color=labels)
