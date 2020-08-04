import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.classification_data import ClassificationData
import models.networks as networks
from util.writer import Writer
from util.util import seg_accuracy, print_network
from models.layers.mesh import Mesh

from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold

activations = dict()

def act_hook(m, input, output, name=None):
    #grab output to layer and store in dict with key as layer name
    print('Inside ' + m.__class__.__name__ + ' forward')
    print('')
    print('input size:', input[0].size())
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

if __name__ == '__main__':
    print('Running hook')
    opt = TrainOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    opt.serial_batches = True  # no shuffle
    dataset = ClassificationData(opt)
    dataloader = DataLoader(dataset,
            batch_size=1,
            shuffle=not opt.serial_batches,
            num_workers=1,
            collate_fn=collate_fn)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                          opt.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
    net.train()
    #optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    #scheduler = networks.get_scheduler(optimizer, opt)
    print_network(net)

    #mesh = Mesh("../data/style/Building/manifold/Asian_1.obj", device=device, hold_history=True)
    #mesh = Mesh(file="../data/style/Building/manifold/Asian_1.obj", opt=opt, hold_history=False, export_folder=opt.export_folder)

    writer = Writer(opt)

    for i in range(3):
        name = 'conv{}'.format(i)
        getattr(net, name).register_forward_hook(partial(act_hook, name=name))

    # test
    oneData = next(iter(dataloader))
    # for i in range(len(oneData['mesh'][0].gemm_edges)):
    #     if -1 in oneData['mesh'][0].gemm_edges[i]:
    #         print(i, oneData['mesh'][0].gemm_edges[i])
    feat, mesh = oneData["edge_features"].to(device), oneData["mesh"]
    net(feat, mesh)

    X = activations["conv0"].squeeze().T
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=50)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.savefig("T-SNE")



    # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(activations["conv0"].squeeze().T)
    # distances, indices = nbrs.kneighbors(activations["conv0"].squeeze()[1].T)
    #
    # mesh1 = trimesh.load_mesh('datasets/shrec_16/alien/test/T124.obj')
    # mesh1.apply_translation([5, 0, 0])
    # mesh2 = trimesh.load_mesh('datasets/shrec_16/alien/test/T411.obj')
    #
    # sc = trimesh.scene.Scene([mesh1, mesh2])
    # mesh2.visual.face_colors = trimesh.visual.interpolate(distances.mean(axis=1), color_map='jet') #.clip(0, 0.3)
    # mesh1.visual.face_colors = 0
    # plt.hist(distances.mean(axis=1))
    # plt.savefig("test")
    # # Show the scene
    # mesh2.show(smooth=False)
    #
    # print("end")
