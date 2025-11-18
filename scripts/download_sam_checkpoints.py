#!/usr/bin/env python3
"""
SAM 模型检查点下载工具
Download SAM Model Checkpoints

自动下载 SAM 和 MedSAM 的预训练模型

使用方法:
    python scripts/download_sam_checkpoints.py --model vit_b --output ./checkpoints
    python scripts/download_sam_checkpoints.py --model all  # 下载所有模型
"""

import argparse
import os
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """显示下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """下载文件并显示进度"""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_sam_checkpoint(model_type: str, output_dir: str):
    """
    下载 SAM 检查点

    Args:
        model_type: 模型类型 ("vit_b", "vit_l", "vit_h")
        output_dir: 输出目录
    """
    # SAM 官方检查点 URL
    checkpoint_urls = {
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "filename": "sam_vit_b_01ec64.pth"
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "filename": "sam_vit_l_0b3195.pth"
        },
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_vit_h_4b8939.pth"
        }
    }

    if model_type not in checkpoint_urls:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(checkpoint_urls.keys())}")

    info = checkpoint_urls[model_type]
    output_path = os.path.join(output_dir, info["filename"])

    # 如果已存在，跳过下载
    if os.path.exists(output_path):
        print(f"✓ Checkpoint already exists: {output_path}")
        return output_path

    # 下载
    os.makedirs(output_dir, exist_ok=True)
    download_url(info["url"], output_path)
    print(f"✓ Downloaded {model_type} checkpoint to {output_path}")

    return output_path


def download_medsam_checkpoint(output_dir: str):
    """
    下载 MedSAM 检查点

    Args:
        output_dir: 输出目录
    """
    # MedSAM 检查点 URL
    medsam_url = "https://zenodo.org/records/10155347/files/medsam_vit_b.pth"
    filename = "medsam_vit_b.pth"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"✓ MedSAM checkpoint already exists: {output_path}")
        return output_path

    os.makedirs(output_dir, exist_ok=True)

    try:
        download_url(medsam_url, output_path)
        print(f"✓ Downloaded MedSAM checkpoint to {output_path}")
    except Exception as e:
        print(f"✗ Failed to download MedSAM: {e}")
        print(f"  Please manually download from: {medsam_url}")
        return None

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download SAM/MedSAM Checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h", "medsam", "all"],
        help="Model type to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )

    args = parser.parse_args()

    print("="*70)
    print("SAM Checkpoint Downloader")
    print("="*70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        # 下载所有模型
        print("\nDownloading all SAM models...")
        for model_type in ["vit_b", "vit_l", "vit_h"]:
            print(f"\n--- Downloading SAM-{model_type.upper()} ---")
            download_sam_checkpoint(model_type, str(output_dir))

        print(f"\n--- Downloading MedSAM ---")
        download_medsam_checkpoint(str(output_dir))

    elif args.model == "medsam":
        download_medsam_checkpoint(str(output_dir))

    else:
        download_sam_checkpoint(args.model, str(output_dir))

    print("\n" + "="*70)
    print("Download completed!")
    print(f"Checkpoints saved to: {output_dir.absolute()}")
    print("="*70)

    # 显示如何使用
    print("\n📝 Usage:")
    print(f"   export SAM_CHECKPOINT_DIR={output_dir.absolute()}")
    print("   python run_hpo.py --config config/sam_hpo_config.yaml")


if __name__ == "__main__":
    main()
