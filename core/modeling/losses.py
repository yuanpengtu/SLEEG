import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv

from mmseg.models import LOSSES

@LOSSES.register_module()
class LikelihoodLoss(nn.Module):

    def __init__(
            self,
            id_index = 0,
            ood_index = 1,
            id_weight = 1.0,
            ood_weight = 1.0,
            loss_weight = 1.0,
            margin = 20.0,
            **kwargs
    ):
        super(LikelihoodLoss, self).__init__()
        self.id_weight = id_weight
        self.ood_weight = ood_weight

        self.id_index = id_index
        self.ood_index = ood_index
        self.loss_weight = loss_weight

        self.margin = margin#kwargs.get("margin", 15.0)

    def forward(
            self,
            ood_map,
            gt_map
    ):

        assert ood_map.shape == gt_map.shape

        batchsize = ood_map.size(0)

        ood_map = ood_map.view(batchsize, -1)
        gt_map = gt_map.view(batchsize, -1)

        id_select_map = (gt_map == self.id_index).float() * self.id_weight
        ood_select_map = (gt_map == self.ood_index).float() * self.ood_weight
        loss = torch.sum(ood_map * id_select_map, dim=1) / (torch.sum(id_select_map, dim=1) + 1e-10) - \
               torch.sum(ood_map * ood_select_map, dim=1) / (torch.sum(ood_select_map, dim=1) + 1e-10) + self.margin

        loss = torch.clamp(loss, min=0)
        loss = torch.mean(loss) * self.loss_weight

        return loss
