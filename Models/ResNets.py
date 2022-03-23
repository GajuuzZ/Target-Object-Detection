"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.
See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation
"""

import torch.nn as nn
#import timm

from torchvision import models
from torchvision.models.resnet import conv1x1, conv3x3


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, dilation=dd,
                               bias=False, kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
    def __init__(self, slug='r50', pretrained=False, modifies_layer=False):
        super().__init__()
        # if not pretrained:
        #     print("Caution, not loading pretrained weights.")

        num_bottleneck_filters = 0
        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        else:
            assert False, "Bad slug: %s" % slug

        """elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048"""

        if modifies_layer:
            self.resnet.inplanes = num_bottleneck_filters // 4  # self.resnet.layer2[-1].conv3.out_channels
            ly3_block = self.resnet.layer3[0]._get_name()
            ly3_num = len(self.resnet.layer3)
            self.resnet.layer3 = self._make_layer(eval(ly3_block), 256, ly3_num,
                                                  stride=1, dilation=2)
            ly4_block = self.resnet.layer4[0]._get_name()
            ly4_num = len(self.resnet.layer4)
            self.resnet.layer4 = self._make_layer(eval(ly4_block), 512, ly4_num,
                                                  stride=1, dilation=4)

        self.outplanes = num_bottleneck_filters

        # Remove Classifier.
        self.resnet.__delattr__('avgpool')
        self.resnet.__delattr__('fc')

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.resnet.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.resnet.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.resnet.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.resnet.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.resnet.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.resnet.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # size = x.size()
        # assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
        #    "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        return enc1, enc2, enc3, enc4

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.resnet.bn1.eval()
            for m in [self.resnet.conv1, self.resnet.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self.resnet, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
