import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS


@MODELS.register_module()
class MSELoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.float()
        return F.mse_loss(reg_score, label)