import torch.nn as nn
from transformers import SwinModel, SwinConfig, SwinPreTrainedModel


class SwinTransformer(SwinPreTrainedModel):
    def __init__(self, image_size, num_classes, ema=False):
        config = SwinConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        super(SwinTransformer, self).__init__(config)
        self.config = config
        self.num_classes = num_classes

        self.swin = SwinModel(config, add_pooling_layer=True, use_mask_token=True)
        # classifier head
        self.classifier = nn.Linear(self.swin.num_features, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        if ema:
            for param in self.swin.parameters():
                param.detach_()
            for param in self.classifier.parameters():
                param.detach_()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.swin(x, bool_masked_pos=token_mask, return_dict=return_dict)
        features = outputs[1]
        logits = self.classifier(features)
        return features, logits
