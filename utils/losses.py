import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)

    @staticmethod
    def cross_entropy(p, q):
        q = F.log_softmax(q, dim=-1)
        loss = torch.sum(p * q, dim=-1)
        return - loss.mean()

    @staticmethod
    def stablize_logits(logits):
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        return logits


    def forward(self, features, labels):
        N = features.size(0)
        device = features.device
        # sim: [N, N]
        sim = self.similarity_f(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        # define the positive pairs by the labels
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(N).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        logits = sim - (1 - logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = self.stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = self.cross_entropy(p, logits)
        return loss


class MultiHeadContrastiveLoss(nn.Module):
    def __init__(self, batch_size, world_size, hidden_dim, temperature, gene_list):
        super(MultiHeadContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.gene_list = gene_list
        self.neg_labels = torch.zeros((batch_size * world_size, len(gene_list)), requires_grad=False).long().cuda()
        self.projectors = [nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64, bias=False)) for _ in gene_list]
        self.criteria = SupConLoss(temperature)

    def forward(self, global_features, pos_features, neg_features, labels):
        """
        global_features: [B, C], features of the global image
        pos_features: [B, C], features of the positive regions
        neg_features: [B, C], features of the negative regions
        labels: [B], labels of the images
        """
        N = self.batch_size * self.world_size
        # gather data from all GPUs
        if self.world_size > 1:
            global_features = torch.cat(GatherLayer.apply(global_features), dim=0)
            pos_features = torch.cat(GatherLayer.apply(pos_features), dim=0)
            neg_features = torch.cat(GatherLayer.apply(neg_features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)
        # reshape as [N, C]
        global_features = global_features.view(N, -1)
        pos_features = pos_features.view(N, -1)
        neg_features = neg_features.view(N, -1)
        labels = labels.view(N, -1)

        # concat the data as [3N, C]
        all_features = torch.cat((global_features, pos_features, neg_features), dim=0)
        all_labels = torch.cat((labels, labels, self.neg_labels), dim=0)

        loss = 0
        for i, _ in enumerate(self.gene_list):
            loss += self.criteria(self.projectors[i](all_features), all_labels[:, i])
        loss /= len(self.gene_list)
        return loss


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
        norm = torch.norm(feature_sim, 2, 1).view(-1, 1) + 1e-5
        feature_sim = feature_sim / norm

        gene_sim = gene.mm(gene.t())
        norm = torch.norm(gene_sim, 2, 1).view(-1, 1) + 1e-5
        gene_sim = gene_sim / norm

        batch_loss = torch.mean((feature_sim - gene_sim) ** 2 / N)
        return batch_loss


class RegionContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, world_size, hidden_dim):
        super(RegionContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64, bias=False),
        )

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
        # include the positive-naive region pairs as the positive pairs
        for i in range(N):
            mask[N + i, 2*N + i] = 1
            mask[2*N + i, N + i] = 1
        return mask

    def forward(self, anchor, pos, neg):
        """
        anchor: global features
        pos: positive region features
        neg: negative region features
        """
        N = self.batch_size * self.world_size
        self.projector = self.projector.cuda(anchor.device)
        if self.world_size > 1:
            anchor = torch.cat(GatherLayer.apply(anchor), dim=0)
            pos = torch.cat(GatherLayer.apply(pos), dim=0)
            neg = torch.cat(GatherLayer.apply(neg), dim=0)
        # z: [3N, C]
        z = torch.cat((anchor, pos, neg), dim=0)
        # project to the contrast space
        z = self.projector(z)

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
