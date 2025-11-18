"""
损失函数定义
Loss Functions for Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss (适合分割任务)

    Dice系数衡量预测和真实标签的重叠程度
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测输出 (B, C, H, W) - logits
            target: 真实标签 (B, H, W) - long tensor

        Returns:
            Dice loss
        """
        # 将logits转换为概率
        pred = F.softmax(pred, dim=1)

        # One-hot编码target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # 计算Dice系数
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 返回1 - Dice作为损失
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (适合类别不平衡)

    关注难分类的样本
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测输出 (B, C, H, W) - logits
            target: 真实标签 (B, H, W) - long tensor

        Returns:
            Focal loss
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # 计算pt
        p = torch.exp(-ce_loss)

        # 计算focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Dice Loss + Focal Loss 组合

    结合两者的优势
    """

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super(DiceFocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


class TverskyLoss(nn.Module):
    """
    Tversky Loss (Dice的泛化版本)

    可以通过alpha和beta调整假阳性和假阴性的惩罚
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测输出 (B, C, H, W) - logits
            target: 真实标签 (B, H, W) - long tensor

        Returns:
            Tversky loss
        """
        # 将logits转换为概率
        pred = F.softmax(pred, dim=1)

        # One-hot编码target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # True Positives, False Positives, False Negatives
        tp = (pred * target_one_hot).sum(dim=(2, 3))
        fp = ((1 - target_one_hot) * pred).sum(dim=(2, 3))
        fn = (target_one_hot * (1 - pred)).sum(dim=(2, 3))

        # Tversky指数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky.mean()


def get_loss_function(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    获取损失函数

    Args:
        loss_name: 损失函数名称
        class_weights: 类别权重 (用于处理类别不平衡)
        **kwargs: 损失函数的额外参数

    Returns:
        损失函数模块
    """
    loss_name = loss_name.lower().replace("-", "").replace("_", "")

    loss_dict = {
        "diceloss": DiceLoss,
        "focalloss": FocalLoss,
        "dicefocalloss": DiceFocalLoss,
        "tverskyloss": TverskyLoss,
        "crossentropy": lambda **kw: nn.CrossEntropyLoss(weight=class_weights),
    }

    if loss_name not in loss_dict:
        available_losses = ", ".join(loss_dict.keys())
        raise ValueError(
            f"Unknown loss: {loss_name}. Available losses: {available_losses}"
        )

    # 创建损失函数
    if loss_name == "crossentropy":
        return loss_dict[loss_name](**kwargs)
    else:
        return loss_dict[loss_name](**kwargs)
