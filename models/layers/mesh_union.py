import torch
from torch.nn import ConstantPad2d


class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        '''
        :param n: number of edges in the mesh
        '''
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
        # mask groups and transpose -> groups is old_N*pooled_N
        self.prepare_groups(features, mask)
        # columns of groups indicate edges to average.
        # multiple pooling can make edge contribute to multiple averages (not just single face)
        # channels*old_N  x  old_N*pooled_N -> channels*pooled_N
        fe = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences
        # pad if the mesh still has less edges than pooling requirement
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe

    def prepare_groups(self, features, mask):
        tensor_mask = torch.from_numpy(mask)
        # mask edges where average doesn't need computing (pooled out)
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        # pad if the mesh still has less edges than pooling requirement
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
