import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from mmaction.registry import MODELS


class MSELoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return F.mse_loss(reg_score, label)


class MAELoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return F.l1_loss(reg_score, label)


class HuberLoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return F.huber_loss(reg_score, label)


class LogCoshLoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return self.log_cosh(reg_score, label)

    @staticmethod
    def log_cosh(pred, true):
        loss = torch.log(torch.cosh(pred - true))
        return torch.mean(loss)


class CosineEmbeddingLoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        index = torch.tensor(1).to(label.device)
        return F.cosine_embedding_loss(reg_score, label, index)


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, reg_score, label):
        noise_var = self.noise_sigma ** 2
        noise_var = noise_var.to(label.device)
        # print(reg_score.shape)
        # reg_score = reg_score.flatten().float()
        # print(reg_score.shape)
        label = label.float()
        return self.bmc_loss(reg_score, label, noise_var)

    @staticmethod
    def bmc_loss(pred, target, noise_var):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, 1].
          target: A float tensor of size [batch, 1].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        logits = - (pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(logits.device))  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss


class FocalMSELoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return self.weighted_focal_mse_loss(reg_score, label)

    @staticmethod
    def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
        loss = (inputs - targets) ** 2
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss


class CCCLoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()

        return 1-self.concordance_correlation_coefficient(reg_score, label)

    def concordance_correlation_coefficient(self, y_pred, y_true, eps=1e-8):
        """Concordance correlation coefficient."""

        # Pearson product-moment correlation coefficients
        cor = torch.corrcoef(torch.stack([y_true, y_pred], dim=0))[0, 1]
        cor = cor if not torch.isnan(cor) else torch.tensor(1.).to(y_true.device)

        # Mean
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        # Variance
        var_true = torch.var(y_true)
        var_pred = torch.var(y_pred)
        # Standard deviation
        sd_true = torch.std(y_true)
        sd_pred = torch.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        ccc = numerator / (denominator + eps)
        return ccc


class MultiClassCCCLoss(nn.Module):
    def forward(self, reg_score, label):
        """
        reg_score: [batch_size, num_class]
        label: [batch_size, num_class]
        """
        num_class = label.shape[-1]
        ccc_loss = torch.tensor(0.).to(label.device)
        for i in range(num_class):
            ccc_loss += (1. - self.concordance_correlation_coefficient(reg_score[:, i], label[:, i]))

        return ccc_loss / num_class

    def concordance_correlation_coefficient(self, y_pred, y_true, eps=1e-8):
        """Concordance correlation coefficient."""

        # Pearson product-moment correlation coefficients
        cor = torch.corrcoef(torch.stack([y_true, y_pred], dim=0))[0, 1]
        cor = cor if not torch.isnan(cor) else torch.tensor(1.).to(y_true.device)

        # Mean
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        # Variance
        var_true = torch.var(y_true)
        var_pred = torch.var(y_pred)
        # Standard deviation
        sd_true = torch.std(y_true)
        sd_pred = torch.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        ccc = numerator / (denominator + eps)
        return ccc


class PCCLoss(nn.Module):

    def forward(self, reg_score, label):
        reg_score = reg_score.flatten().float()
        label = label.flatten().float()
        return 1-self.pearson_correlation_coefficient(reg_score, label)

    def pearson_correlation_coefficient(self, y_pred, y_true):
        """Pearson correlation coefficient."""

        # Pearson product-moment correlation coefficients
        cor = torch.corrcoef(torch.stack([y_true, y_pred], dim=0))[0, 1]
        cor = cor if not torch.isnan(cor) else torch.tensor(1.).to(y_true.device)
        return cor


class MultiClassPCCLoss(nn.Module):
    def forward(self, reg_score, label):
        """
        reg_score: [batch_size, num_class]
        label: [batch_size, num_class]
        """
        num_class = label.shape[-1]
        pcc_loss = torch.tensor(0.).to(label.device)
        for i in range(num_class):
            pcc_loss += (1. - self.pearson_correlation_coefficient(reg_score[:, i], label[:, i]))

        return pcc_loss / num_class

    def pearson_correlation_coefficient(self, y_pred, y_true):
        """Concordance correlation coefficient."""

        # Pearson product-moment correlation coefficients
        cor = torch.corrcoef(torch.stack([y_true, y_pred], dim=0))[0, 1]
        cor = cor if not torch.isnan(cor) else torch.tensor(1.).to(y_true.device)

        return cor


@MODELS.register_module()
class RegressionLoss(nn.Module):
    def __init__(self, loss_type='MSE', loss_weight=None):
        super(RegressionLoss, self).__init__()

        self.loss_type = loss_type.split('+')
        if loss_weight is None:
            self.loss_weight = [1 for _ in range(len(self.loss_type))]
        else:
            self.loss_weight = loss_weight

        self.config = self.config_dict()

    @staticmethod
    def config_dict():
        return dict(
            MSE=MSELoss(),
            MAE=MAELoss(),
            Huber=HuberLoss(),
            LogCosh=LogCoshLoss(),
            Cosine=CosineEmbeddingLoss(),
            BMC=BMCLoss(1.0),
            FMSE=FocalMSELoss(),
            CCC=CCCLoss(),
            MCCC=MultiClassCCCLoss(),
            PCC=PCCLoss(),
            MPCC=MultiClassPCCLoss()
        )

    def forward(self, reg_score, label):
        loss = 0
        for i, lt in enumerate(self.loss_type):
            loss += self.loss_weight[i]*self.config[lt](reg_score, label)
        return loss


class PearsonEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.):
        super(PearsonEmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, target, reduction='mean'):
        batch_size = x1.size(0)
        if len(target) == 1:
            target = target.repeat(batch_size)

        assert x1.shape == x2.shape
        if len(x1.shape) == 3:
            x1 = x1.contiguous().view(-1, x1.size(-1))
            x2 = x2.contiguous().view(-1, x2.size(-1))

        scores = []
        for i in range(batch_size):
            score = self._pearson_similarity(x1[i], x2[i])
            score = self._cal_score(score, target[i].item())
            scores.append(score)
        scores = torch.stack(scores, 0)
        if reduction == 'mean':
            return scores.mean()
        elif reduction == 'sum':
            return scores.sum()

    def _pearson_similarity(self, x, y):
        n = len(x)
        # simple sums
        sum1 = sum(float(x[i]) for i in range(n))
        sum2 = sum(float(y[i]) for i in range(n))
        # sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in x])
        sum2_pow = sum([pow(v, 2.0) for v in y])
        # sum up the products
        p_sum = sum([x[i] * y[i] for i in range(n)])
        # 分子num，分母den
        num = p_sum - (sum1 * sum2 / n)
        den = torch.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if den == 0:
            return 0.0
        return num / den

    def _cal_score(self, score, target):
        if target == 1:
            return 1 - score
        else:
            return max(0, score - self.margin)