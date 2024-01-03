import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_swin_cam(model, images, labels, smooth=True):
    training = model.training
    with torch.no_grad():
        target_layer = model.swin.layernorm
        grad_cam = GradCAM(model=model, target_layers=target_layer)
        targets = [ClassifierOutputTarget(labels)]
        cam = grad_cam(input_tensor=images, targets=targets, eigen_smooth=smooth)
    model.train(training)
    return cam
