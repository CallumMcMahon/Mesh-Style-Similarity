import time
import trimesh
import numpy as np
import pickle
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
from sklearn.cluster import MiniBatchKMeans

activations = dict()

def act_hook(m, input, output, name=None):
    #grab output to layer and store in dict with key as layer name
    print('Inside ' + m.__class__.__name__ + ' forward')
    print('input size:', input[0].size(), "\n")
    activations[name] = output.detach().to("cpu")

def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = ['mesh', 'label', 'style']
    for key in keys:
        meta[key] = np.array([d[key] for d in batch])
    meta['label'] = torch.from_numpy(meta['label']).long()
    meta['style'] = torch.from_numpy(meta['style']).long()
    meta['edge_features'] = pad_sequence([d['edge_features'].float().T for d in batch],
                                         batch_first=True).transpose(1,2)
    return meta

def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    for layer in layers:
        getattr(obj, layer).register_forward_hook(partial(act_hook, name=layer))

if __name__ == '__main__':
    print('Running gram')
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
    # input_nc, ncf, ninput_edges, pool_res, nclasses, opt, gpu_ids, arch, init_type, init_gain, fc_n=100

    # schec
    net = networks.define_classifier(opt.input_nc, [64, 128, 256, 256], opt.ninput_edges, [4000, 3500, 3000, 2500],
                                     1, 30, opt, opt.gpu_ids, 'mconvnet', opt.init_type, opt.init_gain, 100)
    # stateDict = torch.load("model_files/shrec_pretrained.pth")
    # layers = ["conv{}".format(i) for i in range(4)]

    # human
    # net = networks.define_classifier(opt.input_nc, [32, 64, 128, 256], opt.ninput_edges, [6000, 3500, 3000, 2500],
    #                                  3, 8, opt, opt.gpu_ids, 'meshunet', opt.init_type, opt.init_gain)
    # stateDict = torch.load("model_files/human_pretrained.pth")
    # layers = ["encoder.convs.{}".format(i) for i in range(4)] + ["decoder.up_convs.{}".format(i) for i in range(3)]
    #
    # net.train()
    # net.load_state_dict(stateDict, strict=False)
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # print_network(net)

    # register_hooks(net, layers, False)


    # relevant styles and classes
    rel_styles = {"Cabriole":0, "Smooth":1, "Straight":2}
    rel_classes = ["Leg"]
    # classActivations = [[[] for i in range(3)] for j in range(len(layers))]
    # all styles and classes
    print(len(dataset))
    classes, styles = dataset.classes, dataset.styles
    for batch in dataloader:
        print(classes[batch["label"][0]], styles[batch["style"][0]])
        # if classes[batch["label"][0]] in rel_classes:
        #     if styles[batch["style"][0]] in rel_styles.keys():
        #         print("got an item")
        # feat, mesh = batch["edge_features"].to(device), batch["mesh"]
        # mesh[0].init_history()
        # net(feat, mesh)
    #             for i, layer in enumerate(layers):
    #                 classActivations[i][rel_styles[styles[batch["style"][0]]]].append(
    #                     activations[layer][0, :, :mesh[0].edges_count, 0].T)
    # pickle.dump(classActivations, open("visualisations/human_leg_Activations.pkl", "wb"))

    # X = torch.cat(partActivations)
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
    # Y = tsne.fit_transform(X)
    # plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1)
    # plt.savefig("T-SNE4")
    # torch.save(X, "rawDataBaroque.pt")
    # torch.save(Y, "tsneDataBaroque.pt")
    #
    # cluster = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=100,
    #                           n_init=100, max_no_improvement=10, verbose=0)
    #
    # labels = cluster.fit_predict(X)
