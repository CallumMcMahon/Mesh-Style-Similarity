import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers

from models.networks import MResConv, MeshPool, get_norm_layer, get_norm_args
from data.classification_data import ClassificationData
from torch.utils.data import DataLoader


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

    def training_step(self, batch, batch_idx):
        x, meshes, y, styles = batch["edge_features"], batch["mesh"], batch["label"], batch["style"]
        print(x.shape, meshes[0].filename)
        y_hat = self(x, meshes)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, "log": {"train_loss": loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'epoch_loss': epoch_loss},
            'progress_bar': {'epoch_loss': epoch_loss}
        }


def collate_fn(batch):
    """Creates mini-batch tensors, padding features to a single size
    """
    meta = {}
    keys = ['mesh', 'label', 'style']
    for key in keys:
        meta[key] = np.array([d[key] for d in batch])
    meta['label'] = torch.from_numpy(meta['label']).long()
    meta['style'] = torch.from_numpy(meta['style']).long()
    meta['edge_features'] = pad_sequence([d['edge_features'].float().T for d in batch],
                                         batch_first=True).transpose(1, 2)
    return meta


from options.train_options import TrainOptions
opt = TrainOptions().parse()
dataset = ClassificationData(opt)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=collate_fn)

norm_layer = get_norm_layer("group", num_groups=16)
model = LitModel(norm_layer, 5, [32, 64, 128, 256], len(dataset.classes), [7250]+[7000, 6750, 6500, 6250], 100, 3)

class MyPrintingCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[MyPrintingCallback()], logger=tb_logger)
trainer.fit(model, dataloader)