import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MultiHeadContrastiveLoss(nn.Module):
    def __init__(self, batch_size, world_size, hidden_dim, gene_list):
        super(MultiHeadContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.gene_list = gene_list
        self.projectors = [nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=False),
            nn.ReLU(),
        ) for _ in gene_list]
        self.neg_labels = torch.zeros(batch_size * world_size // 2, requires_grad=False).long().cuda()
        self.criteria = SupConLoss()

    def forward(self, global_features, pos_features, neg_features, labels):
        """
        global_features: [B, C], features of the global image
        pos_features: [B, C], features of the positive regions
        neg_features: [B, C], features of the negative regions
        labels: [B], labels of the images
        """
        N = self.batch_size * self.world_size
        assert N % 2 == 0
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
        labels = labels.view(N)

        # construct different views
        view1 = torch.cat((global_features, neg_features[:N//2]))
        view2 = torch.cat((pos_features, neg_features[N//2:]))
        features = torch.stack((view1, view2), dim=1)
        labels = torch.cat((labels, self.neg_labels), dim=0)
        loss = 0
        for i, _ in enumerate(self.gene_list):
            projector = self.projectors[i].cuda(features.device)
            # project to the contrast space
            proj_features = projector(features)
            # compute the contrastive loss
            loss += self.criteria(proj_features, labels)
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
        # project to the contrast space
        anchor, pos, neg = self.projector(anchor), self.projector(pos), self.projector(neg)
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
