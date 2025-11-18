"""
通用工具函数
Utility Functions
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    设置随机种子以保证可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU

    # 确保确定性（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict,
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
):
    """
    保存模型检查点

    Args:
        state: 包含模型状态、优化器状态等的字典
        save_dir: 保存目录
        filename: 文件名
        is_best: 是否是最佳模型
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename
    torch.save(state, filepath)

    if is_best:
        best_path = save_path / "best_model.pth"
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional = None
) -> Dict:
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）

    Returns:
        检查点字典（包含epoch等信息）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def load_config(config_path: str) -> Dict:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """
    获取可用的计算设备

    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数量

    Args:
        model: PyTorch模型

    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """
    早停机制

    当验证指标在patience轮内没有改善时停止训练
    """

    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的epoch数
            mode: 'max' (指标越大越好) 或 'min' (指标越小越好)
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前指标值

        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
