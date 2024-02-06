import torch
import torch.nn as nn


class ContrastiveProjectors(nn.Module):
    def __init__(self, hidden_dim, gene_list, teacher=False):
        super(ContrastiveProjectors, self).__init__()
        self.region_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64, bias=False),
        )
        self.gene_projectors = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim, 64, bias=False),
                nn.ReLU(),
            ) for _ in gene_list]
        )

        if teacher:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, features):
        region_features = self.region_projector(features)
        # num_gene * [B, 64] -> [num_gene, B, 64]
        gene_features = torch.stack([gene_projector(features) for gene_projector in self.gene_projectors], dim=0)
        return region_features, gene_features