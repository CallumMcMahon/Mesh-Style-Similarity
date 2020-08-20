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
from torch.utils.data.sampler import SubsetRandomSampler


class LitModel(pl.LightningModule):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, pool_res, fc_n,
                 nresblocks=3):
        super().__init__()
        self.k = [nf0] + conv_res
        self.res = pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x)).squeeze(-1)
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx, raise_oom=False):
        try:
            x, meshes, label, styles = batch["edge_features"], batch["mesh"], batch["label"], batch["style"]
            # print(x.shape, meshes[0].filename)
            y_hat = self(x, meshes)
            loss = F.cross_entropy(y_hat, label)
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

    def validation_step(self, batch, batch_idx, raise_oom=False):
        try:
            x, meshes, label, styles = batch["edge_features"], batch["mesh"], batch["label"], batch["style"]
            # print(x.shape, meshes[0].filename)
            y_hat = self(x, meshes)
            loss = F.cross_entropy(y_hat, label)
            return {'val_loss': loss}
        except RuntimeError as e:
            if 'out of memory' in str(e) and not raise_oom:
                print('| WARNING: ran out of memory, retrying batch')
                for p in self.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return self.validation_step(batch, batch_idx, raise_oom=True)
            else:
                raise e

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

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

batch_size=1
validation_split = .2
# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trn_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, sampler=valid_sampler)

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

#tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(gpus=1, precision=16, accumulate_grad_batches=int(8/batch_size), callbacks=[MyPrintingCallback()],
                     checkpoint_callback=checkpoint_callback)#, logger=tb_logger)#, amp_level='O1')
trainer.fit(model, trn_loader, val_loader)