
import os
import math
from abc import ABC
import torch
import torch.nn as nn


pretrained_dir = './pretrained'
model_urls = {
    'resnet18': os.path.join(pretrained_dir, 'resnet18-imagenet.pth'),
    'resnet34': os.path.join(pretrained_dir, 'resnet34-imagenet.pth'),
    'resnet50': os.path.join(pretrained_dir, 'resnet50-imagenet.pth'),
    'resnet101': os.path.join(pretrained_dir, 'resnet101-imagenet.pth'),
    'resnet152': os.path.join(pretrained_dir, 'resnet152-imagenet.pth'),
}


def conv_3x3(input_channels, n_filters, stride=1, padding=1):
    return nn.Conv2d(input_channels, n_filters, kernel_size=3, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    """ 残差模块 """
    def __init__(self, input_channels, n_filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_3x3(input_channels, n_filters, stride=stride)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(n_filters, n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.stride = stride
        self.downsample = downsample
        self.expansion = 1

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module, ABC):
    def __init__(self, input_channels, n_filters, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, n_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv3 = nn.Conv2d(n_filters, n_filters * 4, kernel_size=1,  bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.conv1 = conv_3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv_3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, n_filters, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != n_filters * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, n_filters * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_filters * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, n_filters, stride, downsample))
        input_channels = n_filters * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(input_channels, n_filters))

        return nn.Sequential(*layers)


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained and os.path.exists(model_urls['resnet18']):
        model.load_state_dict(torch.load(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained and os.path.exists(model_urls['resnet34']):
        model.load_state_dict(torch.load(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained and os.path.exists(model_urls['resnet50']):
        model.load_state_dict(torch.load(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained and os.path.exists(model_urls['resnet101']):
        model.load_state_dict(torch.load(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained and os.path.exists(model_urls['resnet152']):
        model.load_state_dict(torch.load(model_urls['resnet152']))
    return model
