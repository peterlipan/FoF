import torch.nn as nn
from transformers import Swinv2Model, Swinv2Config, Swinv2PreTrainedModel


class SwinTransformer(Swinv2PreTrainedModel):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4, window_size=7):
        config = Swinv2Config().from_pretrained(pretrained) if pretrained else Swinv2Config()
        config.num_labels = num_classes
        config.image_size = image_size
        if not pretrained:
            config.patch_size = patch_size
            config.window_size = window_size
        super(SwinTransformer, self).__init__(config)
        self.config = config
        self.num_classes = num_classes
        
        self.swin = Swinv2Model(config, add_pooling_layer=True, use_mask_token=True)
        if pretrained:
            self.swin = Swinv2Model.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
        self.classifier = nn.Linear(self.swin.num_features, config.num_labels)     

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.swin(x, bool_masked_pos=token_mask, return_dict=return_dict)
        features = outputs[1]
        logits = self.classifier(features)
        return features, logits
