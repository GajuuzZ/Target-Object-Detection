# Ref. from: https://github.com/HonglinChu/SiamTrackers/blob/570f2ad833b03a1340831e94d050e3b3c2ac3f0e/3-SiamRPN/SiamRPNpp-UP/siamrpnpp/models

import torch
import torch.nn as nn
import torch.nn.functional as F

from XCorr import xcorr_depthwise, xcorr_fast


class UPChannelRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=0):
        super(UPChannelRPN, self).__init__()
        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(in_channels, in_channels * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(in_channels, in_channels * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(nn.Module):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted

        self.rpns = nn.ModuleList()
        for i in range(len(in_channels)):
            self.rpns.append(DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))

        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def _avg(self, lst):
        return sum(lst) / len(lst)

    def _weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for i, (z_f, x_f) in enumerate(zip(z_fs, x_fs)):
            c, l = self.rpns[i](z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            return self._weighted_avg(cls, cls_weight), self._weighted_avg(loc, loc_weight)
        else:
            return self._avg(cls), self._avg(loc)


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        self.downsample = nn.ModuleList()
        for i in range(self.num):
            self.downsample.append(
                AdjustLayer(in_channels[i], out_channels[i], center_size))

    def forward(self, features):
        out = []
        for i in range(self.num):
            out.append(self.downsample[i](features[i]))
        return out
