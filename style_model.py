import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
#from test import run_test
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from data.classification_data import ClassificationData
from models.mesh_classifier import ClassifierModel
from util.writer import Writer

from functools import partial
from sklearn.neighbors import NearestNeighbors
import trimesh
import matplotlib.pyplot as plt
import numpy as np


def act_hook(m, input, output, name=None):
    # grab output to layer and store in dict with key as layer name
    print('Inside ' + m.__class__.__name__ + ' forward')
    print('')
    print('input size:', input[0].size())
    activations[name] = output.to("cpu")

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
    meta['edge_features'] = pad_sequence([torch.from_numpy(d['edge_features']).float().T for d in batch],
                                         batch_first=True).transpose(1,2)
    return meta

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = ClassificationData(opt)
    dataloader = DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    model = ClassifierModel(opt)

    # activations = dict()
    #
    # for i in range(3):
    #     name = 'conv{}'.format(i)
    #     getattr(model.net.module, name).register_forward_hook(partial(act_hook, name=name))
    #
    # i, data = next(enumerate(dataloader))
    # model.set_input(data)
    # ncorrect, nexamples = model.test()

    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)

    writer.close()