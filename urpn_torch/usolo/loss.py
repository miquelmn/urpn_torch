import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.1

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum(axis=0)
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum(axis=0) + y_true.sum(axis=0) + self.smooth
        )
        return 1.0 - dsc


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__loss = DiceLoss()

    def forward(self, mask_pred, cls_pred, mask_true):
        mask_pred = mask_pred[:, :][cls_pred > 0.5]
        mask_true = mask_true[:, :][cls_pred > 0.5]

        loss = self.__loss(mask_pred, mask_true)

        return loss


class SoloLoss(nn.Module):
    def __init__(self, lambd=3):
        super().__init__()

        self.__loss_mask = MaskLoss()
        self.__lambda = lambd

    def forward(self, mask_pred, cls_pred, mask_true, cls_true):
        class_loss = torch.mean(sigmoid_focal_loss(cls_pred, cls_true))
        mask_loss = self.__lambda * self.__loss_mask(mask_pred, cls_pred, mask_true)
        loss = class_loss + mask_loss

        return loss
