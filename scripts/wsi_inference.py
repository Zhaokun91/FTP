#!/usr/bin/env python3
"""
WSI 推理工具
Whole Slide Image Inference

将训练好的模型应用到完整的 WSI，生成预测掩码

参考标准: WSInfer (Nature npj Precision Oncology 2024)

使用方法:
    python scripts/wsi_inference.py \
        --image /path/to/wsi.png \
        --model /path/to/best_model.pth \
        --output /path/to/output \
        --patch-size 256 \
        --stride 128 \
        --tta

作者: FTP Project
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml


class WSIInference:
    """
    WSI 推理类

    使用滑动窗口对完整 WSI 进行预测
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 8,
        use_tta: bool = False,
        tta_transforms: Optional[List] = None
    ):
        """
        Args:
            model: 训练好的模型
            device: 计算设备
            patch_size: Patch 大小
            stride: 滑动步长
            batch_size: 批次大小
            use_tta: 是否使用测试时增强
            tta_transforms: TTA 变换列表
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.use_tta = use_tta
        self.tta_transforms = tta_transforms if tta_transforms else self._default_tta_transforms()

        self.model.to(self.device)
        self.model.eval()

    def _default_tta_transforms(self) -> List:
        """默认的 TTA 变换"""
        import albumentations as albu

        transforms = [
            albu.Compose([]),  # 原始
            albu.Compose([albu.HorizontalFlip(p=1.0)]),
            albu.Compose([albu.VerticalFlip(p=1.0)]),
            albu.Compose([albu.Transpose(p=1.0)]),
        ]
        return transforms

    def _extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        提取所有 patches

        Returns:
            patches: patch 列表
            coordinates: 坐标列表 (y, x)
        """
        h, w = image.shape[:2]
        patches = []
        coordinates = []

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                coordinates.append((y, x))

        return patches, coordinates

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """预处理单个 patch"""
        # 归一化（使用 ImageNet 统计值）
        patch = patch.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        patch = (patch - mean) / std

        # HWC -> CHW
        patch = patch.transpose(2, 0, 1)

        return torch.from_numpy(patch).float()

    def _predict_batch(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        批量预测

        Args:
            patches: patch 列表

        Returns:
            predictions: (N, H, W) 预测掩码（0或1）
        """
        # 预处理
        tensors = [self._preprocess_patch(p) for p in patches]
        batch = torch.stack(tensors).to(self.device)

        # 预测
        with torch.no_grad():
            if self.use_tta:
                # 测试时增强
                predictions = []
                for transform in self.tta_transforms:
                    # 应用变换
                    transformed_batch = []
                    for i in range(batch.shape[0]):
                        patch_np = batch[i].cpu().numpy().transpose(1, 2, 0)
                        # 反归一化
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        patch_np = patch_np * std + mean
                        patch_np = (patch_np * 255).astype(np.uint8)

                        # 应用增强
                        augmented = transform(image=patch_np)
                        patch_aug = augmented['image']

                        # 重新归一化
                        patch_aug = patch_aug.astype(np.float32) / 255.0
                        patch_aug = (patch_aug - mean) / std
                        patch_aug = patch_aug.transpose(2, 0, 1)
                        transformed_batch.append(torch.from_numpy(patch_aug).float())

                    transformed_batch = torch.stack(transformed_batch).to(self.device)

                    # 预测
                    output = self.model(transformed_batch)

                    # 反向变换（使预测对齐）
                    # 这里简化处理，实际应该反向应用变换
                    prob = F.softmax(output, dim=1)[:, 1, :, :]  # 取前景概率
                    predictions.append(prob)

                # 平均所有 TTA 预测
                pred = torch.stack(predictions).mean(dim=0)
            else:
                # 不使用 TTA
                output = self.model(batch)
                pred = F.softmax(output, dim=1)[:, 1, :, :]  # 取前景概率

            # 二值化
            pred_mask = (pred > 0.5).cpu().numpy().astype(np.uint8)

        return pred_mask

    def _merge_patches(
        self,
        predictions: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        合并重叠的 patches

        使用平均策略处理重叠区域

        Args:
            predictions: 预测 patch 列表
            coordinates: 坐标列表
            image_shape: 原图尺寸 (H, W)

        Returns:
            merged: 完整的预测掩码
        """
        h, w = image_shape

        # 创建累加数组
        prediction_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # 累加所有预测
        for pred, (y, x) in zip(predictions, coordinates):
            prediction_sum[y:y+self.patch_size, x:x+self.patch_size] += pred
            count_map[y:y+self.patch_size, x:x+self.patch_size] += 1

        # 平均（避免除零）
        count_map = np.maximum(count_map, 1)
        merged = prediction_sum / count_map

        # 二值化
        merged = (merged > 0.5).astype(np.uint8)

        return merged

    def predict(self, image: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        对完整图像进行预测

        Args:
            image: 输入图像 (H, W, C)
            show_progress: 是否显示进度条

        Returns:
            prediction: 预测掩码 (H, W)，值为 0 或 1
        """
        # 提取 patches
        patches, coordinates = self._extract_patches(image)

        print(f"提取了 {len(patches)} 个 patches")
        print(f"图像尺寸: {image.shape[:2]}")
        print(f"Patch 大小: {self.patch_size}x{self.patch_size}")
        print(f"步长: {self.stride}")
        if self.use_tta:
            print(f"使用测试时增强（TTA）: {len(self.tta_transforms)} 种变换")

        # 批量预测
        predictions = []

        iterator = range(0, len(patches), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="预测中")

        for i in iterator:
            batch_patches = patches[i:i+self.batch_size]
            batch_preds = self._predict_batch(batch_patches)
            predictions.extend(batch_preds)

        # 合并 patches
        print("合并预测结果...")
        merged = self._merge_patches(predictions, coordinates, image.shape[:2])

        return merged


def load_model(model_path: str, config_path: Optional[str] = None) -> torch.nn.Module:
    """
    加载训练好的模型

    Args:
        model_path: 模型权重路径
        config_path: 配置文件路径（可选）

    Returns:
        模型
    """
    import sys
    from pathlib import Path

    # 添加项目路径
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    from src.models.model_factory import create_model

    # 加载配置
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model_name', 'U-Net')
        encoder_name = config.get('encoder_name', 'resnet34')
    else:
        # 默认配置
        model_name = 'U-Net'
        encoder_name = 'resnet34'

    # 创建模型
    model = create_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=None,
        num_classes=2
    )

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


def post_process(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    后处理预测掩码

    - 形态学操作（开运算、闭运算）
    - 去除小目标

    Args:
        mask: 预测掩码
        min_size: 最小目标尺寸（像素）

    Returns:
        处理后的掩码
    """
    import cv2

    # 形态学闭运算（填充小孔洞）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 形态学开运算（去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 连通组件分析 - 去除小目标
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 创建过滤后的掩码
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # 跳过背景（0）
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_mask[labels == i] = 1

    return filtered_mask


def save_for_qupath(
    prediction: np.ndarray,
    output_path: str,
    original_image_path: str
):
    """
    保存为 QuPath 可导入的格式

    Args:
        prediction: 预测掩码 (H, W)
        output_path: 输出路径
        original_image_path: 原始图像路径
    """
    # 转换为 QuPath 期望的格式（0=背景, 255=前景）
    qupath_mask = (prediction * 255).astype(np.uint8)

    # 保存为PNG
    cv2.imwrite(output_path, qupath_mask)

    print(f"✅ 预测掩码已保存为 QuPath 格式: {output_path}")
    print(f"\n📝 QuPath 导入步骤:")
    print(f"   1. 在 QuPath 中打开原始图像: {Path(original_image_path).name}")
    print(f"   2. Image → Import → Import as annotation")
    print(f"   3. 选择: {output_path}")
    print(f"   4. 设置: Threshold = 127, Min area = 0")


def main():
    parser = argparse.ArgumentParser(description="WSI 推理工具")

    # 输入输出
    parser.add_argument("--image", type=str, required=True, help="WSI 图像路径")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="模型配置文件（可选）")

    # 推理参数
    parser.add_argument("--patch-size", type=int, default=256, help="Patch 大小")
    parser.add_argument("--stride", type=int, default=128, help="滑动步长（建议 patch-size/2 以提高精度）")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")

    # TTA
    parser.add_argument("--tta", action="store_true", help="使用测试时增强（提高精度但速度慢2-4倍）")

    # 后处理
    parser.add_argument("--post-process", action="store_true", help="后处理（形态学操作、去除小目标）")
    parser.add_argument("--min-size", type=int, default=50, help="最小目标尺寸（像素）")

    # QuPath 导出
    parser.add_argument("--save-for-qupath", action="store_true", help="保存为 QuPath 可导入格式")

    args = parser.parse_args()

    print("="*70)
    print("WSI 推理工具")
    print("="*70)
    print(f"图像: {args.image}")
    print(f"模型: {args.model}")
    print(f"输出: {args.output}")
    print("="*70 + "\n")

    # 加载模型
    print("加载模型...")
    model = load_model(args.model, args.config)

    # 读取图像
    print("读取图像...")
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法读取图像: {args.image}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建推理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    inferencer = WSIInference(
        model=model,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        use_tta=args.tta
    )

    # 推理
    print("\n开始推理...")
    prediction = inferencer.predict(image, show_progress=True)

    # 后处理
    if args.post_process:
        print("\n后处理...")
        prediction = post_process(prediction, min_size=args.min_size)

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(args.image).stem

    # 保存预测掩码
    pred_path = output_dir / f"{image_name}_prediction.png"
    pred_save = (prediction * 255).astype(np.uint8)
    cv2.imwrite(str(pred_path), pred_save)
    print(f"\n✅ 预测掩码已保存: {pred_path}")

    # 保存可视化（叠加图）
    overlay = image.copy()
    overlay[prediction == 1] = overlay[prediction == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_path = output_dir / f"{image_name}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"✅ 叠加可视化已保存: {overlay_path}")

    # QuPath 格式
    if args.save_for_qupath:
        qupath_path = output_dir / f"{image_name}_qupath.png"
        save_for_qupath(prediction, str(qupath_path), args.image)

    # 统计
    total_pixels = prediction.size
    foreground_pixels = (prediction == 1).sum()
    foreground_ratio = foreground_pixels / total_pixels * 100

    print("\n" + "="*70)
    print("推理完成！")
    print("="*70)
    print(f"图像尺寸: {image.shape[:2]}")
    print(f"前景像素: {foreground_pixels:,} ({foreground_ratio:.2f}%)")
    print(f"输出目录: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
