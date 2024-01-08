import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable


class GeneGuidance(nn.Module):
    def __init__(self, batch_size, world_size):
        super(GeneGuidance, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
    
    def forward(self, features, gene):
        N = self.batch_size * self.world_size
        # gather data from all GPUs
        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            gene = torch.cat(GatherLayer.apply(gene), dim=0)
        # reshape as NxC
        features = features.view(N, -1)
        gene = gene.view(N, -1)

        # sample-wise relationship, NxN
        feature_sim = features.mm(features.t())
        norm = torch.norm(feature_sim, 2, 1).view(-1, 1)
        feature_sim = feature_sim / norm

        gene_sim = gene.mm(gene.t())
        norm = torch.norm(gene_sim, 2, 1).view(-1, 1)
        gene_sim = gene_sim / norm

        batch_loss = torch.mean((feature_sim - gene_sim) ** 2 / N)
        return batch_loss


class RegionContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, world_size, dataparallel):
        super(RegionContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.dataparallel = dataparallel

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = batch_size * world_size
        sub_mask = torch.ones((2*N, 2*N), dtype=bool)
        sub_mask = sub_mask.fill_diagonal_(0)
        # zero out the positive pairs
        for i in range(N):
            sub_mask[i, N + i] = 0
            sub_mask[N + i, i] = 0

        mask = torch.zeros((3*N, 3*N), dtype=bool)
        mask[:2*N, :2*N] = sub_mask
        # include the global-negative region pairs as the negative pairs
        for i in range(N):
            mask[i, 2*N + i] = 1
            mask[2*N + i, i] = 1
        return mask

    def forward(self, anchor, pos, neg):
        """
        anchor: global features
        pos: positive region features
        neg: negative region features
        """
        N = self.batch_size * self.world_size

        if self.world_size > 1:
            anchor = torch.cat(GatherLayer.apply(anchor), dim=0)
            pos = torch.cat(GatherLayer.apply(pos), dim=0)
            neg = torch.cat(GatherLayer.apply(neg), dim=0)
        # z: [3N, D]
        z = torch.cat((anchor, pos, neg), dim=0)
        # sim: [3N, 3N]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # positive pairs: anchor-pos
        sim_i_j = torch.diag(sim, 2*N)[:N]
        sim_j_i = torch.diag(sim, -2*N)[:N]

        positive_pairs = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*N, 1)
        negative_pairs = sim[self.mask].reshape(2*N, -1)

        labels = torch.zeros(2*N).to(positive_pairs.device).long()
        logits = torch.cat((positive_pairs, negative_pairs), dim=1)
        loss = self.criterion(logits, labels)
        loss /= (2*N)
        return loss
