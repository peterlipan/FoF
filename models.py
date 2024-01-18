import torch.nn as nn
from transformers import SwinModel, SwinConfig, SwinPreTrainedModel


class SwinTransformer(SwinPreTrainedModel):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4, window_size=7):
        config = SwinConfig().from_pretrained(pretrained) if pretrained else SwinConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        config.window_size = window_size
        super(SwinTransformer, self).__init__(config)
        self.config = config
        self.num_classes = num_classes
        
        self.swin = SwinModel(config, add_pooling_layer=True, use_mask_token=True)
        if pretrained:
            self.swin = SwinModel.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
        self.classifier = nn.Linear(self.swin.num_features, config.num_labels)     

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.swin(x, bool_masked_pos=token_mask, return_dict=return_dict)
        features = outputs[1]
        logits = self.classifier(features)
        return features, logits
