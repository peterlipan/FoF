import torch
from functools import partial
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        features, logits = self.model(x)
        return logits


def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_swin_cam(model, images, labels, smooth=True):
    training = model.training
    with torch.no_grad():
        target_layer = model.swin.layernorm
        reshape_transform = partial(swinT_reshape_transform_huggingface,
                            width=images.shape[3]//32,
                            height=images.shape[2]//32)
        grad_cam = ScoreCAM(model=HuggingfaceToTensorModelWrapper(model), target_layers=[target_layer], reshape_transform=reshape_transform)
        targets = [ClassifierOutputTarget(labels)]
        cam = grad_cam(input_tensor=images, targets=targets, eigen_smooth=smooth)
    model.train(training)
    return cam
