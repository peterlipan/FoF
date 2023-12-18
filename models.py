import torchvision
import torch.nn as nn


class CreateModel(nn.Module):
    def __init__(self, backbone, image_size, num_classes, hid_dim=256, pretrained=True, ema=False):
        super(CreateModel, self).__init__()
        encoder = getattr(torchvision.models, backbone)(image_size=image_size,pretrained=pretrained)
        self.num_classes = num_classes

        in_features = encoder.heads.head.in_features
        # use MLP as the classification head
        encoder.heads = nn.Sequential(
            nn.Linear(in_features, hid_dim),
            nn.Tanh(),
        )
        classifier = nn.Linear(hid_dim, num_classes)

        if ema:
            for param in encoder.parameters():
                param.detach_()
            for param in classifier.parameters():
                param.detach_()

        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        features = self.encoder(x)
        pred = self.classifier(features)
        return features, pred
