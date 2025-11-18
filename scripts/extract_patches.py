#!/usr/bin/env python3
"""
WSI 切片工具
Patch Extraction from Whole Slide Images and Masks

将完整的 WSI 图像和对应的掩码切成小块（patches）用于训练

使用方法:
    # 基础用法
    python scripts/extract_patches.py \
        --image-dir /path/to/wsi_images \
        --mask-dir /path/to/wsi_masks \
        --output-dir /path/to/output/Patches \
        --patch-size 256

    # 高级用法
    python scripts/extract_patches.py \
        --image-dir /path/to/wsi_images \
        --mask-dir /path/to/wsi_masks \
        --output-dir /path/to/output/Patches \
        --patch-size 512 \
        --stride 256 \
        --min-foreground 0.1 \
        --background-value 0

作者: FTP Project
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
from tqdm import tqdm
import shutil


def extract_patches_from_image(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = None,
    min_foreground_ratio: float = 0.05,
    background_value: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    从大图中提取 patches

    Args:
        image: 输入图像 (H, W, C)
        mask: 对应的掩码 (H, W)
        patch_size: patch 大小
        stride: 滑动步长（None 则使用 patch_size，即无重叠）
        min_foreground_ratio: 最小前景比例（过滤空白 patch）
        background_value: 背景值（通常为 0）

    Returns:
        image_patches: 图像 patches 列表
        mask_patches: 掩码 patches 列表
        coordinates: 每个 patch 的坐标 (y, x)
    """
    if stride is None:
        stride = patch_size

    h, w = image.shape[:2]
    image_patches = []
    mask_patches = []
    coordinates = []

    # 滑动窗口提取
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # 提取 patch
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # 检查前景比例
            foreground_pixels = (mask_patch != background_value).sum()
            foreground_ratio = foreground_pixels / (patch_size * patch_size)

            # 只保留包含足够前景的 patch
            if foreground_ratio >= min_foreground_ratio:
                image_patches.append(img_patch)
                mask_patches.append(mask_patch)
                coordinates.append((y, x))

    return image_patches, mask_patches, coordinates


def process_wsi_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    patch_size: int = 256,
    stride: int = None,
    min_foreground_ratio: float = 0.05,
    background_value: int = 0,
    save_format: str = 'png'
) -> int:
    """
    处理一对 WSI 图像和掩码

    Args:
        image_path: WSI 图像路径
        mask_path: WSI 掩码路径
        output_image_dir: 输出图像目录
        output_mask_dir: 输出掩码目录
        patch_size: patch 大小
        stride: 滑动步长
        min_foreground_ratio: 最小前景比例
        background_value: 背景值
        save_format: 保存格式 ('png', 'jpg', 'tif')

    Returns:
        提取的 patch 数量
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️  无法读取图像: {image_path}")
        return 0

    # 读取掩码
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️  无法读取掩码: {mask_path}")
        return 0

    # 检查尺寸是否匹配
    if image.shape[:2] != mask.shape[:2]:
        print(f"⚠️  图像和掩码尺寸不匹配:")
        print(f"   图像: {image.shape[:2]}")
        print(f"   掩码: {mask.shape[:2]}")
        return 0

    # 提取 patches
    image_patches, mask_patches, coordinates = extract_patches_from_image(
        image=image,
        mask=mask,
        patch_size=patch_size,
        stride=stride,
        min_foreground_ratio=min_foreground_ratio,
        background_value=background_value
    )

    # 保存 patches
    base_name = image_path.stem
    for idx, (img_patch, mask_patch, (y, x)) in enumerate(zip(image_patches, mask_patches, coordinates)):
        # 生成文件名
        patch_name = f"{base_name}_y{y:05d}_x{x:05d}_idx{idx:04d}.{save_format}"

        # 保存图像 patch
        img_save_path = output_image_dir / patch_name
        cv2.imwrite(str(img_save_path), img_patch)

        # 保存掩码 patch
        mask_save_path = output_mask_dir / patch_name
        cv2.imwrite(str(mask_save_path), mask_patch)

    return len(image_patches)


def main():
    parser = argparse.ArgumentParser(description="从 WSI 提取训练 patches")

    # 输入输出路径
    parser.add_argument("--image-dir", type=str, required=True, help="WSI 图像目录")
    parser.add_argument("--mask-dir", type=str, required=True, help="WSI 掩码目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录（会创建 images 和 labels 子文件夹）")

    # Patch 参数
    parser.add_argument("--patch-size", type=int, default=256, help="Patch 大小（默认: 256）")
    parser.add_argument("--stride", type=int, default=None, help="滑动步长（默认: 等于 patch-size，无重叠）")
    parser.add_argument("--min-foreground", type=float, default=0.05,
                       help="最小前景比例，过滤空白 patch（默认: 0.05 = 5%%）")
    parser.add_argument("--background-value", type=int, default=0,
                       help="背景像素值（默认: 0）")

    # 其他选项
    parser.add_argument("--save-format", type=str, default="png", choices=["png", "jpg", "tif"],
                       help="保存格式（默认: png）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")

    args = parser.parse_args()

    # 创建路径对象
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    # 检查输入目录
    if not image_dir.exists():
        print(f"❌ 图像目录不存在: {image_dir}")
        return

    if not mask_dir.exists():
        print(f"❌ 掩码目录不存在: {mask_dir}")
        return

    # 检查输出目录
    if output_dir.exists():
        if args.overwrite:
            print(f"⚠️  删除已存在的输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"❌ 输出目录已存在: {output_dir}")
            print(f"   使用 --overwrite 覆盖，或选择其他输出目录")
            return

    # 创建输出目录
    output_image_dir = output_dir / "images"
    output_mask_dir = output_dir / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("WSI Patch 提取工具")
    print("="*70)
    print(f"图像目录: {image_dir}")
    print(f"掩码目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n参数:")
    print(f"  Patch 大小: {args.patch_size}x{args.patch_size}")
    print(f"  滑动步长: {args.stride if args.stride else args.patch_size}")
    print(f"  最小前景比例: {args.min_foreground*100:.1f}%")
    print(f"  背景值: {args.background_value}")
    print(f"  保存格式: {args.save_format}")
    print("="*70 + "\n")

    # 获取图像文件列表
    image_files = sorted([f for f in image_dir.glob("*")
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])

    if not image_files:
        print(f"❌ 在 {image_dir} 中没有找到图像文件")
        return

    print(f"找到 {len(image_files)} 个 WSI 图像\n")

    # 处理每对图像和掩码
    total_patches = 0
    processed_files = 0
    failed_files = []

    for image_file in tqdm(image_files, desc="处理 WSI"):
        # 查找对应的掩码文件
        mask_file = mask_dir / image_file.name

        # 尝试不同扩展名
        if not mask_file.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                mask_file = mask_dir / f"{image_file.stem}{ext}"
                if mask_file.exists():
                    break

        if not mask_file.exists():
            print(f"\n⚠️  找不到对应的掩码: {image_file.name}")
            failed_files.append(image_file.name)
            continue

        # 处理这对文件
        num_patches = process_wsi_pair(
            image_path=image_file,
            mask_path=mask_file,
            output_image_dir=output_image_dir,
            output_mask_dir=output_mask_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            min_foreground_ratio=args.min_foreground,
            background_value=args.background_value,
            save_format=args.save_format
        )

        if num_patches > 0:
            total_patches += num_patches
            processed_files += 1
            tqdm.write(f"  ✓ {image_file.name}: 提取 {num_patches} 个 patches")
        else:
            failed_files.append(image_file.name)

    # 总结
    print("\n" + "="*70)
    print("提取完成！")
    print("="*70)
    print(f"处理成功: {processed_files}/{len(image_files)} 个 WSI")
    print(f"提取 patches 总数: {total_patches}")
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── images/  ({len(list(output_image_dir.glob('*')))} 个文件)")
    print(f"  └── labels/  ({len(list(output_mask_dir.glob('*')))} 个文件)")

    if failed_files:
        print(f"\n⚠️  失败的文件 ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files)-10} 个")

    print("\n下一步:")
    print(f"  1. 验证数据: python scripts/validate_data.py --data-dir {output_dir}")
    print(f"  2. 开始训练: python run_hpo.py --config config/hpo_config.yaml")
    print("="*70)


if __name__ == "__main__":
    main()
