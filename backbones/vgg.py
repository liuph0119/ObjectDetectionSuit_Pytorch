
import os
import torch
import torch.nn as nn


pretrained_dir = './pretrained'
model_urls = {
    'vgg16': os.path.join(pretrained_dir, 'vgg16-imagenet.pth'),
    'vgg19': os.path.join(pretrained_dir, 'vgg19-imagenet.pth'),
}


class VGG(nn.Module):
    def __init__(self, input_channels=3, layers=None, num_class=1000):
        super(VGG, self).__init__()

    def forward(self, x):
        return x


def vgg16(pretrained=False, **kwargs):
    model = VGG(input_channels=3, layers=[2, 2, 4, 4], **kwargs)
    if pretrained and os.path.exists(model_urls['vgg16']):
        model.load_state_dict(torch.load(model_urls['vgg16']))
    return model


def vgg19(pretrained=False, **kwargs):
    model = VGG(input_channels=3, layers=[2, 2, 4, 4], **kwargs)
    if pretrained and os.path.exists(model_urls['vgg19']):
        model.load_state_dict(torch.load(model_urls['vgg19']))
    return model
