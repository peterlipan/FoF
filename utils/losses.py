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
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
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

        x = F.log_softmax(feature_sim, dim=1)
        target = F.softmax(gene_sim, dim=1)
        loss = self.criterion(x, target)
        return loss


