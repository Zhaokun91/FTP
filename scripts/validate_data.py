#!/usr/bin/env python3
"""
数据格式验证脚本
Validate Data Format

验证数据集中的掩码格式是否正确：
- 背景 = 0
- 骨细胞 = 1

使用方法:
    python scripts/validate_data.py --data-dir /path/to/Patches
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm


def validate_mask_format(data_dir: str, num_samples: int = 10):
    """
    验证掩码格式

    Args:
        data_dir: 数据目录
        num_samples: 验证的样本数量
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    mask_dir = data_path / "labels"

    # 检查目录
    if not image_dir.exists():
        print(f"❌ 图像目录不存在: {image_dir}")
        return False

    if not mask_dir.exists():
        print(f"❌ 掩码目录不存在: {mask_dir}")
        return False

    # 获取文件列表
    image_files = list(image_dir.glob("*"))
    mask_files = list(mask_dir.glob("*"))

    print("="*70)
    print("数据格式验证")
    print("="*70)
    print(f"图像目录: {image_dir}")
    print(f"掩码目录: {mask_dir}")
    print(f"图像数量: {len(image_files)}")
    print(f"掩码数量: {len(mask_files)}")
    print()

    # 验证样本
    samples_to_check = min(num_samples, len(mask_files))
    print(f"验证前 {samples_to_check} 个掩码文件...")
    print()

    all_values = []
    issues = []

    for i, mask_file in enumerate(tqdm(mask_files[:samples_to_check], desc="验证中")):
        # 读取掩码
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            issues.append(f"❌ 无法读取: {mask_file.name}")
            continue

        # 统计像素值
        unique_values = np.unique(mask)
        all_values.extend(unique_values.tolist())

        # 检查是否只有 0 和 1（或 0 和 255）
        if not (set(unique_values) <= {0, 1} or set(unique_values) <= {0, 255}):
            issues.append(
                f"⚠️  {mask_file.name}: "
                f"包含非预期值 {unique_values}"
            )

    # 统计所有像素值
    value_counts = Counter(all_values)

    print("\n" + "="*70)
    print("验证结果")
    print("="*70)

    print(f"\n📊 掩码中的像素值分布:")
    for value, count in sorted(value_counts.items()):
        print(f"   值 {value:3d}: 出现在 {count} 个文件中")

    print(f"\n✅ 预期格式:")
    print(f"   - 背景: 0")
    print(f"   - 骨细胞: 1 或 255")

    # 判断格式
    print(f"\n🔍 格式检查:")
    unique_all = set(all_values)

    if unique_all <= {0, 1}:
        print(f"   ✅ 格式正确！掩码使用 0 (背景) 和 1 (骨细胞)")
        format_ok = True
    elif unique_all <= {0, 255}:
        print(f"   ✅ 格式可接受！掩码使用 0 (背景) 和 255 (骨细胞)")
        print(f"   ℹ️  系统会自动转换为 0 和 1")
        format_ok = True
    else:
        print(f"   ❌ 格式异常！检测到非预期值: {unique_all}")
        format_ok = False

    # 问题报告
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... 还有 {len(issues)-10} 个问题")
    else:
        print(f"\n✅ 未发现问题")

    print("\n" + "="*70)

    return format_ok


def show_sample(data_dir: str, sample_id: str = None):
    """
    显示样本图像和掩码

    Args:
        data_dir: 数据目录
        sample_id: 样本ID（文件名，不含扩展名）
    """
    import matplotlib.pyplot as plt

    data_path = Path(data_dir)
    image_dir = data_path / "images"
    mask_dir = data_path / "labels"

    # 获取样本
    if sample_id is None:
        # 随机选择一个
        image_files = list(image_dir.glob("*"))
        if not image_files:
            print("❌ 没有找到图像文件")
            return
        sample_file = image_files[0]
        sample_id = sample_file.stem
    else:
        sample_file = list(image_dir.glob(f"{sample_id}.*"))[0]

    # 查找对应的掩码
    mask_file = list(mask_dir.glob(f"{sample_id}.*"))[0]

    # 读取
    image = cv2.imread(str(sample_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    # 显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title(f"图像: {sample_id}")
    axes[0].axis('off')

    # 掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"掩码 (原始值: {np.unique(mask)})")
    axes[1].axis('off')

    # 二值化后的掩码
    mask_binary = (mask > 127).astype(np.uint8)
    axes[2].imshow(mask_binary, cmap='gray')
    axes[2].set_title(f"二值化掩码 (0: 背景, 1: 骨细胞)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('data_validation_sample.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 样本可视化已保存到: data_validation_sample.png")

    # 显示统计
    print(f"\n📊 样本统计:")
    print(f"   图像形状: {image.shape}")
    print(f"   掩码形状: {mask.shape}")
    print(f"   掩码像素值: {np.unique(mask)}")
    print(f"   骨细胞像素数: {(mask_binary == 1).sum()}")
    print(f"   背景像素数: {(mask_binary == 0).sum()}")
    print(f"   骨细胞占比: {(mask_binary == 1).sum() / mask_binary.size * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="验证数据格式")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="数据目录路径（包含 images 和 labels 子文件夹）"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="验证的样本数量（默认: 10）"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="显示一个样本图像和掩码"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="指定要显示的样本ID"
    )

    args = parser.parse_args()

    # 验证格式
    format_ok = validate_mask_format(args.data_dir, args.num_samples)

    # 显示样本
    if args.show_sample:
        print("\n" + "="*70)
        show_sample(args.data_dir, args.sample_id)

    # 总结
    print("\n" + "="*70)
    if format_ok:
        print("✅ 数据格式验证通过！可以开始训练。")
    else:
        print("❌ 数据格式存在问题，请检查掩码文件。")
    print("="*70)


if __name__ == "__main__":
    main()
