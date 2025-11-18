#!/usr/bin/env python3
"""
Patch 预览工具
Preview Extracted Patches

在提取 patches 之前，预览将会生成多少 patches，以及它们的分布

使用方法:
    python scripts/preview_patches.py \
        --image /path/to/wsi.png \
        --mask /path/to/mask.png \
        --patch-size 256 \
        --stride 256
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def preview_patch_extraction(
    image_path: str,
    mask_path: str,
    patch_size: int = 256,
    stride: int = None,
    min_foreground_ratio: float = 0.05,
    max_preview_patches: int = 16
):
    """
    预览 patch 提取效果

    Args:
        image_path: 图像路径
        mask_path: 掩码路径
        patch_size: patch 大小
        stride: 滑动步长
        min_foreground_ratio: 最小前景比例
        max_preview_patches: 最大预览数量
    """
    if stride is None:
        stride = patch_size

    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]

    print("="*70)
    print("Patch 提取预览")
    print("="*70)
    print(f"图像尺寸: {w} x {h}")
    print(f"Patch 大小: {patch_size} x {patch_size}")
    print(f"滑动步长: {stride}")
    print(f"最小前景比例: {min_foreground_ratio*100:.1f}%")
    print()

    # 计算可能的 patch 位置
    valid_patches = []
    all_patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            foreground_ratio = (mask_patch > 0).sum() / (patch_size * patch_size)

            all_patches.append((y, x, foreground_ratio))

            if foreground_ratio >= min_foreground_ratio:
                valid_patches.append((y, x, foreground_ratio))

    print(f"总 patch 数（所有位置）: {len(all_patches)}")
    print(f"有效 patch 数（前景 >= {min_foreground_ratio*100:.1f}%）: {len(valid_patches)}")
    print(f"过滤比例: {(1 - len(valid_patches)/len(all_patches))*100:.1f}%")
    print()

    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 原始图像 + patch 位置标注
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.imshow(image)
    ax1.set_title(f"原始图像 ({w}x{h})", fontsize=14)
    ax1.axis('off')

    # 绘制所有 patch 边界（半透明）
    for y, x, ratio in all_patches[:500]:  # 限制绘制数量
        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                  linewidth=0.5, edgecolor='blue',
                                  facecolor='none', alpha=0.3)
        ax1.add_patch(rect)

    # 绘制有效 patch 边界（高亮）
    for y, x, ratio in valid_patches[:500]:
        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                  linewidth=1, edgecolor='green',
                                  facecolor='green', alpha=0.1)
        ax1.add_patch(rect)

    # 2. 掩码 + patch 位置标注
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.imshow(mask, cmap='gray')
    ax2.set_title(f"掩码（背景=0，骨细胞=1或255）", fontsize=14)
    ax2.axis('off')

    # 绘制有效 patch
    for y, x, ratio in valid_patches[:500]:
        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                  linewidth=1, edgecolor='red',
                                  facecolor='red', alpha=0.2)
        ax2.add_patch(rect)

    # 3. 前景比例分布直方图
    ax3 = fig.add_subplot(gs[1, :2])
    ratios = [r for _, _, r in all_patches]
    ax3.hist(ratios, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(min_foreground_ratio, color='red', linestyle='--',
               label=f'阈值: {min_foreground_ratio*100:.1f}%')
    ax3.set_xlabel('前景比例', fontsize=12)
    ax3.set_ylabel('Patch 数量', fontsize=12)
    ax3.set_title('前景比例分布', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 统计信息
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')

    stats_text = f"""
    📊 统计信息

    原始图像尺寸: {w} × {h} = {w*h/1e6:.2f} M 像素

    Patch 设置:
      • Patch 大小: {patch_size} × {patch_size}
      • 滑动步长: {stride}
      • 重叠: {(patch_size - stride) if stride < patch_size else 0} 像素

    提取结果:
      • 总 patch 数: {len(all_patches)}
      • 有效 patch: {len(valid_patches)} ({len(valid_patches)/len(all_patches)*100:.1f}%)
      • 过滤掉: {len(all_patches) - len(valid_patches)} ({(1-len(valid_patches)/len(all_patches))*100:.1f}%)

    前景统计:
      • 平均前景比例: {np.mean(ratios)*100:.2f}%
      • 中位数: {np.median(ratios)*100:.2f}%
      • 最大: {np.max(ratios)*100:.2f}%

    预估训练数据量:
      • 图像数据: {len(valid_patches) * patch_size * patch_size * 3 / 1e6:.1f} MB
      • 掩码数据: {len(valid_patches) * patch_size * patch_size / 1e6:.1f} MB
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 5. 预览一些有效 patches
    preview_count = min(max_preview_patches, len(valid_patches))
    if preview_count > 0:
        # 随机选择一些 patches
        indices = np.linspace(0, len(valid_patches)-1, preview_count, dtype=int)

        for idx, patch_idx in enumerate(indices):
            row = 2 + idx // 6
            col = idx % 6

            if row >= 3:
                break

            y, x, ratio = valid_patches[patch_idx]
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # 创建叠加视图
            overlay = img_patch.copy()
            overlay[mask_patch > 0] = overlay[mask_patch > 0] * 0.5 + np.array([255, 0, 0]) * 0.5

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(overlay.astype(np.uint8))
            ax.set_title(f'前景: {ratio*100:.1f}%', fontsize=8)
            ax.axis('off')

    # 保存和显示
    output_path = 'patch_extraction_preview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 预览图已保存到: {output_path}")
    print()
    print("="*70)

    return len(valid_patches)


def main():
    parser = argparse.ArgumentParser(description="预览 Patch 提取效果")
    parser.add_argument("--image", type=str, required=True, help="WSI 图像路径")
    parser.add_argument("--mask", type=str, required=True, help="WSI 掩码路径")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch 大小")
    parser.add_argument("--stride", type=int, default=None, help="滑动步长")
    parser.add_argument("--min-foreground", type=float, default=0.05, help="最小前景比例")

    args = parser.parse_args()

    num_patches = preview_patch_extraction(
        image_path=args.image,
        mask_path=args.mask,
        patch_size=args.patch_size,
        stride=args.stride if args.stride else args.patch_size,
        min_foreground_ratio=args.min_foreground
    )

    print(f"\n💡 预估:")
    print(f"   如果处理所有 WSI，将生成约 {num_patches} 个 patches")
    print(f"   建议调整参数以获得合适的数据量（推荐: 500-5000 patches）")


if __name__ == "__main__":
    main()
