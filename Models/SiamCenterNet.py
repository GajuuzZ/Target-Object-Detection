import torch
import torch.nn as nn

from Models.ResNets import ResNet
from Models.RPN import F, DepthwiseXCorr, AdjustAllLayer


class DepthwiseRPN_Center(nn.Module):
    """SiameseRPN head with CenterNet output. """
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseRPN_Center, self).__init__()

        self.hm_head = DepthwiseXCorr(in_channels, hidden, out_channels,
                                      kernel_size=kernel_size)

        self.wh_head = DepthwiseXCorr(in_channels, hidden, 2,
                                      kernel_size=kernel_size)
        self.reg_head = DepthwiseXCorr(in_channels, hidden, 2,
                                       kernel_size=kernel_size)

    def forward(self, zf, xf):
        hm = self.hm_head(zf, xf)
        wh = self.wh_head(zf, xf)
        offset = self.reg_head(zf, xf)
        return hm, wh, offset


class MultiRPN_Center(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, weighted=False):
        super(MultiRPN_Center, self).__init__()
        self.weighted = weighted

        self.rpns = nn.ModuleList()
        for i in range(len(in_channels)):
            self.rpns.append(DepthwiseRPN_Center(in_channels[i], hidden, out_channels))

        if self.weighted:
            self.hm_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.wh_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.offset_weight = nn.Parameter(torch.ones(len(in_channels)))

    def _avg(self, lst):
        return sum(lst) / len(lst)

    def _weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def forward(self, zfs, xfs):
        hm_l = []
        wh_l = []
        offset_l = []
        for i, (zf, xf) in enumerate(zip(zfs, xfs)):
            hm, wh, offset = self.rpns[i](zf, xf)
            hm_l.append(hm)
            wh_l.append(wh)
            offset_l.append(offset)

        if self.weighted:
            hm_w = F.softmax(self.hm_weight, 0)
            wh_w = F.softmax(self.wh_weight, 0)
            offset_w = F.softmax(self.offset_weight, 0)
            return self._weighted_avg(hm_l, hm_w), self._weighted_avg(wh_l, wh_w), \
                   self._weighted_avg(offset_l, offset_w)
        else:
            return self._avg(hm_l), self._avg(wh_l), self._avg(offset_l)


class SiamCenterNet_ResNet(nn.Module):
    def __init__(self, slug='r50'):
        super(SiamCenterNet_ResNet, self).__init__()

        self.backbone = ResNet(slug, pretrained=False, modifies_layer=True)
        layer2_out = self.backbone.resnet.layer2[-1].expansion * 128
        layer3_out = self.backbone.resnet.layer3[-1].expansion * 256
        layer4_out = self.backbone.resnet.layer4[-1].expansion * 512
        self.neck = AdjustAllLayer([layer2_out, layer3_out, layer4_out],
                                   [256, 256, 256])
        self.rpn = MultiRPN_Center([256, 256, 256], 256, 1, True)

    def forward(self, template, search):
        zf = self.backbone(template)
        xf = self.backbone(search)

        zf = self.neck(zf[1:])
        xf = self.neck(xf[1:])

        hm, wh, offset = self.rpn(zf, xf)
        hm = torch.sigmoid(hm)
        wh = torch.sigmoid(wh)
        offset = torch.sigmoid(offset)
        return hm, wh, offset

    def get_output_shape(self, template_shape, search_shape):
        device = self.backbone.resnet.conv1.weight.device
        with torch.no_grad():
            template = torch.zeros(template_shape, dtype=torch.float32).to(device)
            search = torch.zeros(search_shape, dtype=torch.float32).to(device)

            hm, wh, offset = self.forward(template, search)

        return hm.shape, wh.shape, offset.shape


if __name__ == '__main__':
    model = SiamCenterNet_ResNet('r50').cuda()

    SEARCH_SIZE = (255, 255)
    TEMPLATE_SIZE = (127, 127)

    t_img = torch.rand((4, 3, *TEMPLATE_SIZE)).cuda()
    s_img = torch.rand((4, 3, *SEARCH_SIZE)).cuda()

    hm, wh, offset = model(t_img, s_img)




