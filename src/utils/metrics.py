"""
评估指标
Evaluation Metrics for Segmentation
"""

import torch
import torch.nn.functional as F
from typing import Dict


class DiceScore:
    """
    Dice系数计算器

    Dice系数衡量预测和真实标签的重叠程度
    范围: [0, 1]，越接近1越好
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice系数

        Args:
            pred: 预测输出 (B, C, H, W) - logits
            target: 真实标签 (B, H, W) - long tensor

        Returns:
            平均Dice系数
        """
        # 将logits转换为类别预测
        pred = torch.argmax(pred, dim=1)

        # 初始化dice分数
        dice_scores = []

        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        # 返回平均Dice（通常只关注前景类）
        return torch.stack(dice_scores).mean()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测输出 (B, C, H, W) - logits
        target: 真实标签 (B, H, W) - long tensor
        num_classes: 类别数

    Returns:
        平均IoU
    """
    # 将logits转换为类别预测
    pred = torch.argmax(pred, dim=1)

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.stack(ious).mean()


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算像素准确率

    Args:
        pred: 预测输出 (B, C, H, W) - logits
        target: 真实标签 (B, H, W) - long tensor

    Returns:
        像素准确率
    """
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().float()
    total = target.numel()
    return correct / total


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred: 预测输出 (B, C, H, W) - logits
        target: 真实标签 (B, H, W) - long tensor
        num_classes: 类别数

    Returns:
        包含所有指标的字典
    """
    dice_calculator = DiceScore(num_classes=num_classes)

    metrics = {
        'dice': dice_calculator(pred, target).item(),
        'iou': compute_iou(pred, target, num_classes).item(),
        'pixel_accuracy': compute_pixel_accuracy(pred, target).item()
    }

    return metrics


class AverageMeter:
    """
    平均值计算器（用于跟踪训练指标）
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
