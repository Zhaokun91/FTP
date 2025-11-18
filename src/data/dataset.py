"""
数据集定义模块
Dataset Definition Module
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    """
    分割任务的数据集类

    Args:
        image_dir: 图像文件夹路径
        mask_dir: 掩码文件夹路径
        image_ids: 图像ID列表（文件名）
        transform: 数据增强函数
        preprocessing: 预处理函数
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_ids: List[str],
        transform: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_ids = image_ids
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本

        Returns:
            image: 图像张量 (C, H, W)
            mask: 掩码张量 (H, W)
        """
        # 加载图像和掩码
        image_id = self.image_ids[idx]

        # 支持多种图像格式
        image_path = self._find_image_path(self.image_dir, image_id)
        mask_path = self._find_image_path(self.mask_dir, image_id)

        # 读取图像 (RGB)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码 (灰度)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 二值化掩码 (0: 背景, 1: 前景)
        mask = (mask > 127).astype(np.uint8)

        # 应用数据增强
        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 预处理（归一化等）
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 转换为张量
        if not isinstance(image, torch.Tensor):
            # HWC -> CHW
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _find_image_path(self, directory: Path, image_id: str) -> Path:
        """查找图像文件，支持多种扩展名"""
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

        # 如果image_id已包含扩展名
        if Path(image_id).suffix:
            return directory / image_id

        # 否则尝试常见扩展名
        for ext in extensions:
            path = directory / f"{image_id}{ext}"
            if path.exists():
                return path

        raise FileNotFoundError(f"Image {image_id} not found in {directory}")


def prepare_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_seed: int = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    准备训练和验证数据加载器

    Args:
        data_dir: 数据根目录（应包含 images 和 labels 子文件夹）
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
        random_seed: 随机种子
        train_transform: 训练集增强
        val_transform: 验证集增强

    Returns:
        train_loader, val_loader
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    mask_dir = data_path / "labels"

    # 检查目录是否存在
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {mask_dir}")

    # 获取所有图像ID（去除扩展名）
    image_files = list(image_dir.glob("*"))
    image_ids = [f.stem for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]

    if len(image_ids) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # 划分训练集和验证集
    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=val_split,
        random_state=random_seed,
        shuffle=True
    )

    print(f"Dataset split: {len(train_ids)} training, {len(val_ids)} validation")

    # 创建数据集
    train_dataset = SegmentationDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        image_ids=train_ids,
        transform=train_transform
    )

    val_dataset = SegmentationDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        image_ids=val_ids,
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
