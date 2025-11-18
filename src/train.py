"""
训练函数模块
Training Functions with W&B Integration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import wandb

from .utils.metrics import DiceScore, compute_metrics, AverageMeter
from .utils.utils import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    log_wandb: bool = True
) -> Dict[str, float]:
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch
        use_amp: 是否使用混合精度训练
        log_wandb: 是否记录到W&B

    Returns:
        包含训练指标的字典
    """
    model.train()

    loss_meter = AverageMeter()
    dice_calculator = DiceScore(num_classes=2)
    dice_meter = AverageMeter()

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 计算Dice
        with torch.no_grad():
            dice = dice_calculator(outputs, masks)

        # 更新指标
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(dice.item(), batch_size)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'dice': f'{dice_meter.avg:.4f}'
        })

        # 记录到W&B（每N个batch记录一次）
        if log_wandb and batch_idx % 10 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_dice': dice.item(),
                'train/epoch': epoch
            })

    metrics = {
        'train_loss': loss_meter.avg,
        'train_dice': dice_meter.avg
    }

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_wandb: bool = True
) -> Dict[str, float]:
    """
    验证模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前epoch
        log_wandb: 是否记录到W&B

    Returns:
        包含验证指标的字典
    """
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 计算指标
            metrics = compute_metrics(outputs, masks, num_classes=2)

            # 更新指标
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(metrics['dice'], batch_size)
            iou_meter.update(metrics['iou'], batch_size)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}',
                'iou': f'{iou_meter.avg:.4f}'
            })

    val_metrics = {
        'val_loss': loss_meter.avg,
        'val_dice': dice_meter.avg,
        'val_iou': iou_meter.avg
    }

    # 记录到W&B
    if log_wandb:
        wandb.log({
            'validation/loss': val_metrics['val_loss'],
            'validation/dice': val_metrics['val_dice'],
            'validation/iou': val_metrics['val_iou'],
            'validation/epoch': epoch
        })

    return val_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    use_amp: bool = True,
    early_stopping_patience: int = 10,
    log_wandb: bool = True
) -> Dict[str, float]:
    """
    完整的训练流程

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        use_amp: 是否使用混合精度训练
        early_stopping_patience: 早停容忍轮数
        log_wandb: 是否记录到W&B

    Returns:
        最佳验证指标
    """
    best_dice = 0.0
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)

        # 训练
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            log_wandb=log_wandb
        )

        # 验证
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_wandb=log_wandb
        )

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_dice'])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            if log_wandb:
                wandb.log({'learning_rate': current_lr, 'epoch': epoch})

        # 打印结果
        print(f"Train - Loss: {train_metrics['train_loss']:.4f}, Dice: {train_metrics['train_dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, Dice: {val_metrics['val_dice']:.4f}, IoU: {val_metrics['val_iou']:.4f}")

        # 保存最佳模型
        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            if log_wandb:
                wandb.run.summary['best_dice'] = best_dice
            print(f"✓ Best model updated! Dice: {best_dice:.4f}")

        # 早停检查
        if early_stopping(val_metrics['val_dice']):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    return {
        'best_dice': best_dice,
        'final_train_loss': train_metrics['train_loss'],
        'final_val_loss': val_metrics['val_loss']
    }
