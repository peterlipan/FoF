import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class Transformer(nn.Module):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4):
        super(Transformer, self).__init__()
        config = ViTConfig().from_pretrained(pretrained) if pretrained else ViTConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        
        self.config = config
        self.num_classes = num_classes
        
        self.encoder = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
        if pretrained:
            self.enocer = ViTModel.from_pretrained(pretrained, config=config, add_pooling_layer=False, use_mask_token=True, ignore_mismatched_sizes=True)
        # replace the official ViT 'pooler' as the real pooling layer
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.global_classifier = nn.Linear(config.hidden_size, num_classes)
        self.local_classifier = nn.Linear(config.hidden_size, num_classes + 10)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.encoder(x, bool_masked_pos=token_mask, return_dict=return_dict)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.transpose(1, 2)[:, :, 1:])
        pooled_output = torch.flatten(pooled_output, 1)
        global_logits = self.global_classifier(pooled_output)
        local_logits = self.local_classifier(pooled_output)
        return pooled_output, global_logits, local_logits


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