import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models.networks import MResConv, MeshPool, get_norm_layer, get_norm_args
from data.classification_data import ClassificationData, collate_fn
from torch.utils.data import DataLoader

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

class LitModel(pl.LightningModule):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, pool_res, fc_n,
                 nresblocks=3, content_x_mesh):
        super().__init__()
        self.k = [nf0] + conv_res
        self.res = pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])
        self.content_losses = nn.ModuleList()
        self.style_losses = nn.ModuleList()

        self.model = nn.ModuleDict()
        for i, ki in enumerate(self.k[:-1]):
            self.model.update({'conv{}'.format(i): MResConv(ki, self.k[i + 1], nresblocks)})
            if i == 1:
                target = self(content_x_mesh).detach()
                content_loss = ContentLoss(target)
                self.model.update({"content_loss_{}".format(i): content_loss})
                self.content_losses.append(content_loss)
            self.model('norm{}'.format(i), norm_layer(**norm_args[i]))
            self.model('pool{}'.format(i), MeshPool(self.res[i + 1]))

        target = self(content_img).detach()
        content_loss = ContentLoss(target)
        self.add_module("content_loss_{}".format(i), content_loss)
        self.content_losses.append(content_loss)
        self
        self.style_losses.append()

    def forward(self, x, mesh):

        for name, module in self.model.items():
            if any(meshLayer in name for meshLayer in ["conv", "pool"]):
                x = module(x, mesh)
            else:
                x = module(x).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx, raise_oom=False):
        try:
            x, meshes, y, styles = batch["edge_features"], batch["mesh"], batch["label"], batch["style"]
            # print(x.shape, meshes[0].filename)
            y_hat = self(x, meshes)
            loss = F.cross_entropy(y_hat, y)
            return {"loss": loss, "log": {"train_loss": loss}}
        except RuntimeError as e:
            if 'out of memory' in str(e) and not raise_oom:
                print('| WARNING: ran out of memory, retrying batch')
                for p in self.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return self.training_step(batch, batch_idx, raise_oom=True)
            else:
                raise e

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'epoch_loss': epoch_loss},
            'progress_bar': {'epoch_loss': epoch_loss}
        }


from options.train_options import TrainOptions
opt = TrainOptions().parse()
dataset = ClassificationData(opt)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=8,
                        collate_fn=collate_fn)

norm_layer = get_norm_layer("group", num_groups=16)
model = LitModel(norm_layer, 5, [32, 64, 128, 256], len(dataset.classes), [7250]+[7000, 6750, 6500, 6250], 100, 3)

class MyPrintingCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=5,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(gpus=1, precision=16, accumulate_grad_batches=4, callbacks=[MyPrintingCallback()],
                     checkpoint_callback=checkpoint_callback, logger=tb_logger, amp_level='O1')
trainer.fit(model, dataloader)