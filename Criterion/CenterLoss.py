import torch.nn as nn
import torch.nn.functional as F

from losses import modified_focal_loss, diou_loss


class CenterLoss(nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=1.):
        super(CenterLoss, self).__init__()
        self.alpha = alpha  # heatmap loss weight
        self.beta = beta  # width-height loss weight
        self.gamma = gamma  # xy-offset loss weight

        self.focal_loss = modified_focal_loss
        # self.iou_loss = diou_loss
        self.l1_loss = F.l1_loss

    def forward(self, pd, gt):
        pd_hm, pd_wh, pd_offset = pd
        gt_hm, gt_wh, gt_offset, gt_ct = gt

        cls_loss = self.focal_loss(pd_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        for b in range(pd_hm.size(0)):
            ct = gt_ct[b]

            pos_pd_wh = pd_wh[b, :, ct[1], ct[0]]
            pos_pd_offset = pd_offset[b, :, ct[1], ct[0]]

            wh_loss += self.l1_loss(pos_pd_wh, gt_wh[b], reduction='sum')
            offset_loss += self.l1_loss(pos_pd_offset, gt_offset[b], reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (b + 1)
