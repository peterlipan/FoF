import torch
import torch.nn as nn
from transformers import Swinv2Model, Swinv2Config, Swinv2PreTrainedModel


class SwinTransformer(nn.Module):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4, window_size=7):
        super(SwinTransformer, self).__init__()
        config = Swinv2Config().from_pretrained(pretrained) if pretrained else Swinv2Config()
        config.num_labels = num_classes
        config.image_size = image_size
        if not pretrained:
            config.patch_size = patch_size
            config.window_size = window_size
        
        self.config = config
        self.num_classes = num_classes
        
        self.swin = Swinv2Model(config, add_pooling_layer=True, use_mask_token=True)
        if pretrained:
            self.swin = Swinv2Model.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
        self.global_classifier = nn.Linear(self.swin.num_features, config.num_labels)
        self.local_classifier = nn.Linear(self.swin.num_features, config.num_labels + 4)     

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.swin(x, bool_masked_pos=token_mask, return_dict=return_dict)
        features = outputs[1]
        global_logits = self.global_classifier(features)
        local_logitis = self.local_classifier(features)
        return features, global_logits, local_logitis


class ContrastiveProjectors(nn.Module):
    def __init__(self, hidden_dim, gene_list):
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
    
    def forward(self, features):
        region_features = self.region_projector(features)
        # num_gene * [B, 64] -> [num_gene, B, 64]
        gene_features = torch.stack([gene_projector(features) for gene_projector in self.gene_projectors], dim=0)
        return region_features, gene_features