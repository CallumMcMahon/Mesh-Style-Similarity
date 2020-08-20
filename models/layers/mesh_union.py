import torch
from torch.nn import ConstantPad2d
import numpy as np

class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        self.groups[target, :] += self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        fe = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe.T

    def prepare_groups(self, features, mask):
        tensor_mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)


class MeshUnionSparse:
    def __init__(self, n, device=torch.device('cpu')):
        '''
        :param n: number of edges in the mesh
        '''
        self.__size = n
        self.device = device
        self.rebuild_features = self.rebuild_features_average
        v = torch.ones(n, device=device)
        i = torch.stack([torch.arange(n, dtype=torch.long, device=device), torch.arange(n, dtype=torch.long, device=device)])
        self.groups = torch.sparse.FloatTensor(i, v)
        #self.groups = torch.eye(n, device=device).to_sparse()

    def union(self, source, target):
        # source row in sparce format
        row = self.groups[source]
        # recreate row in target row of an empty tensor of size groups
        new_row = torch.sparse.FloatTensor(torch.stack([torch.ones(row._indices().shape, dtype=torch.long).to(self.device)*target, row._indices()]).squeeze(1),
                                           row._values(), self.groups.shape)
        self.groups.add_(new_row)

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key]

    def get_occurrences(self):
        return torch.sparse.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask]

    def rebuild_features_average(self, features, mask, target_edges):
        # mask groups and transpose -> groups is old_N*pooled_N
        self.prepare_groups(features, mask)
        # columns of groups indicate edges to average.
        # multiple pooling can make edge contribute to multiple averages (not just single face)
        # channels*old_N  x  old_N*pooled_N -> channels*pooled_N
        fe = torch.sparse.mm(self.groups, features.squeeze(-1).T)
        occurrences = torch.sparse.sum(self.groups, 1).to_dense().unsqueeze(1)
        fe = fe / occurrences
        # pad if the mesh has less edges than pooling requirement
        padding_b = target_edges - fe.shape[0]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, 0, 0, padding_b), 0)
            fe = padding_b(fe)
        return fe

    def in1D(self, labels):
        # modified from https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor
        x = self.groups._indices()[0]
        mapping = torch.zeros(x.size()).byte().to(self.device)
        for label in labels:
            mapping = mapping | x.eq(label)
        return mapping

    def prepare_groups(self, features, mask):
        # mask edges where average doesn't need computing (pooled out)
        # complicated indexing since sparce tensors don't implement boolean masks yet
        tensor_mask = torch.from_numpy(mask)
        mask_idx = (tensor_mask == True).nonzero().to(self.device)
        old2newPos = torch.cumsum(tensor_mask, dim=0) - 1
        v_idx = self.in1D(mask_idx)
        v = torch.clamp(self.groups._values()[v_idx], 0, 1)
        i = self.groups._indices()[:, v_idx]
        i[0] = old2newPos[i[0]]
        # pad if pre-pooled mesh was smaller than previous pooling
        if features.shape[1] - self.groups.shape[1] > 0:
            self.groups = torch.sparse.FloatTensor(i, v, torch.Size([sum(tensor_mask), features.shape[1]]))
        else:
            self.groups = torch.sparse.FloatTensor(i, v, torch.Size([sum(tensor_mask), self.groups.shape[1]]))
