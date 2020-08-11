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
from sklearn.cluster import MiniBatchKMeans

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
    net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                          opt.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
    net.train()
    stateDict = torch.load("checkpoints/shrec16/latest_net.pth")
    del stateDict["fc2.weight"]
    del stateDict["fc2.bias"]
    net.load_state_dict(stateDict, strict=False)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    #optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    #scheduler = networks.get_scheduler(optimizer, opt)
    print_network(net)

    #mesh = Mesh("../data/style/Building/manifold/Asian_1.obj", device=device, hold_history=True)
    #mesh = Mesh(file="../data/style/Building/manifold/Asian_1.obj", opt=opt, hold_history=False, export_folder=opt.export_folder)

    # writer = Writer(opt)


    ######running on a gpu
    # for i in range(3):
    #     name = 'conv{}'.format(i)
    #     getattr(net.module, name).register_forward_hook(partial(act_hook, name=name))

    ######running on a cpu
    for i in range(3):
        name = 'conv{}'.format(i)
        getattr(net, name).register_forward_hook(partial(act_hook, name=name))

    # oneData = dataloader[0]
    # feat, mesh = oneData["edge_features"].to(device), oneData["mesh"]
    # net(feat, mesh)
    #
    # X = activations["conv0"].squeeze().T
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
    # Y = tsne.fit_transform(X)
    # plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1)
    # plt.savefig("T-SNE3")
    partActivations = []
    partDataset = subMeshDataset(opt)
    data = partDataset[22]
    partMesh, label, style = data['partMesh'], data['label'], data['style']
    parts = len(partMesh.sub_mesh)
    minibatch_size = 1
    for i in range(int(parts/minibatch_size)):
        batch = []
        for j in range(minibatch_size):
            mesh = partMesh[j + minibatch_size*i]
            batch.append({'mesh': mesh, 'edge_features': mesh.features, 'label': label, 'style': style})
        batch = collate_fn(batch)
        feat, mesh = batch["edge_features"].to(device), batch["mesh"]
        net(feat, mesh)
        for j in range(minibatch_size):
            partActivations.append(activations["conv0"][j, :, :partMesh[j + minibatch_size*i].edges_count, 0].T)

    X = torch.cat(partActivations)
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.1)
    plt.savefig("T-SNE4")
    torch.save(X, "rawDataBaroque.pt")
    torch.save(Y, "tsneDataBaroque.pt")

    cluster = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=100,
                              n_init=100, max_no_improvement=10, verbose=0)

    labels = cluster.fit_predict(X)


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
