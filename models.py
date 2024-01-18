import torch.nn as nn
from transformers import SwinModel, SwinConfig, SwinPreTrainedModel


class SwinTransformer(SwinPreTrainedModel):
    def __init__(self, image_size, num_classes, ema=False, pretrained="", ema_decay=0.999, patch_size=4, window_size=7):
        config = SwinConfig().from_pretrained(pretrained) if pretrained else SwinConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        config.window_size = window_size
        super(SwinTransformer, self).__init__(config)
        self.config = config
        self.num_classes = num_classes
        self.ema = ema
        self.ema_decay = ema_decay

        if not ema:
            if pretrained:
                self.swin = SwinModel.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
            else:
                self.swin = SwinModel(config, add_pooling_layer=True, use_mask_token=True)
            self.classifier = nn.Linear(self.swin.num_features, config.num_labels)
        
        if ema:
            if pretrained:
                self.global_encoder = SwinModel.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
                self.local_encoder = SwinModel.from_pretrained(pretrained, config=config, add_pooling_layer=True, use_mask_token=True)
            else:
                self.global_encoder = SwinModel(config, add_pooling_layer=True, use_mask_token=True)
                self.local_encoder = SwinModel(config, add_pooling_layer=True, use_mask_token=True)
            self.global_classifier = nn.Linear(self.global_encoder.num_features, config.num_labels)
            # Add a dummy class for the local classifier
            self.local_classifier = nn.Linear(self.local_encoder.num_features, config.num_labels + 1)        

        # Initialize weights and apply final processing
        self.post_init()

    def update_ema_variables(self, step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (step + 1), self.ema_decay)
        for ema_param, param in zip(self.global_encoder.parameters(), self.local_encoder.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def forward(self, x, token_mask=None, global_process=True):
        return_dict = self.config.use_return_dict
        # if it is the teacher model, train the classifier only
        # keep the encoder trainable to calculate the grads in grad-cam
        if self.ema:
            if global_process:
                encoder = self.global_encoder
                classifier = self.global_classifier
            else:
                encoder = self.local_encoder
                classifier = self.local_classifier
        else:
            encoder = self.swin
            classifier = self.classifier

        outputs = encoder(x, bool_masked_pos=token_mask, return_dict=return_dict)
        features = outputs[1]
        logits = classifier(features)
        return features, logits
