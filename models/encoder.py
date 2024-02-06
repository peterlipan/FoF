import torch
import torch.nn as nn
from .MyViT import ViTModel, ViTConfig


class Transformer(nn.Module):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4):
        super(Transformer, self).__init__()
        config = ViTConfig().from_pretrained(pretrained) if pretrained else ViTConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        
        self.config = config
        self.num_classes = num_classes
        
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True, use_cls_token=False)
        if pretrained:
            self.vit = ViTModel.from_pretrained(pretrained, config=config, add_pooling_layer=False, use_mask_token=True, use_cls_token=False, ignore_mismatched_sizes=True)
        # replace the official ViT 'pooler' as the real pooling layer
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.global_classifier = nn.Linear(config.hidden_size, num_classes)
        self.local_classifier = nn.Linear(config.hidden_size, num_classes + 10)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.vit(x, bool_masked_pos=token_mask, return_dict=return_dict)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.transpose(1, 2)[:, :, 1:])
        pooled_output = torch.flatten(pooled_output, 1)
        global_logits = self.global_classifier(pooled_output)
        local_logits = self.local_classifier(pooled_output)
        return pooled_output, global_logits, local_logits
