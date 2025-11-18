# 🦴 骨细胞分割 - 超参数自动优化系统

# Osteoblast Segmentation with Hyperparameter Optimization (HPO)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Sweeps-yellow.svg)](https://wandb.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个强大的**自动化超参数优化系统**，用于医学图像分割任务。系统使用 **Weights & Biases (W&B) Sweeps** 自动搜索最佳超参数组合，无需手动调参！

---

## 📋 目录

- [核心特性](#核心特性)
- [什么是超参数优化 (HPO)](#什么是超参数优化-hpo)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [推理和部署](#推理和部署)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

---

## ✨ 核心特性

### 🚀 自动化超参数优化

- **智能搜索**: 使用贝叶斯优化自动寻找最佳超参数组合
- **并行实验**: 支持同时运行多个实验（多GPU环境）
- **实时监控**: W&B 仪表板实时显示训练进度和结果
- **自动早停**: 防止过拟合，节省计算资源

### 🎯 丰富的模型选择

支持多种先进的分割模型：

**传统 CNN 模型：**
- U-Net, U-Net++, FPN, DeepLabV3+, PSPNet, LinkNet

**🆕 Transformer 模型（SAM 系列）：**
- **SAM (Segment Anything Model)** - Meta AI 的通用分割模型
- **MedSAM** - 专门为医学图像优化的 SAM 变体
- 支持 ViT-B, ViT-L, ViT-H 三种规模

### 📊 多种编码器骨干网络

- ResNet, EfficientNet, DenseNet, MobileNet等

### 🔧 灵活的损失函数

- Dice Loss, Focal Loss, Tversky Loss, 组合损失

---

## 🤔 什么是超参数优化 (HPO)?

### 传统方式（手动调参）❌

```python
# 第1次尝试: lr=1e-4 → Dice=0.80
# 第2次尝试: lr=5e-5 → Dice=0.82
# 😓 重复数十次，既慢又繁琐
```

### HPO 方式（自动优化）✅

```yaml
# 定义搜索空间
learning_rate: {min: 1e-5, max: 1e-3}
batch_size: [8, 16, 32]

# 系统自动运行 50 次实验
# 🎉 最终报告：最佳组合 lr=3e-5, bs=16 → Dice=0.85
```

---

## 📁 项目结构

```
FTP/
├── config/                      # 配置文件
│   ├── hpo_config.yaml         # 完整HPO配置
│   ├── sam_hpo_config.yaml     # SAM专用配置
│   └── sam_vs_unet_config.yaml # SAM对比配置
├── src/                         # 源代码
│   ├── data/                   # 数据处理
│   ├── models/                 # 模型定义（含SAM适配器）
│   ├── utils/                  # 工具函数
│   └── train.py                # 训练逻辑
├── docs/                        # 文档
│   ├── DATA_PREPARATION.md     # 数据准备指南
│   ├── INFERENCE_GUIDE.md      # 推理部署指南
│   └── SAM_USAGE.md            # SAM使用指南
├── notebooks/                   # Jupyter Notebooks
│   └── Osteoblast_Segmentation_HPO.ipynb
├── scripts/                     # 辅助脚本
│   ├── extract_patches.py      # Patch提取
│   ├── preview_patches.py      # Patch预览
│   ├── validate_data.py        # 数据验证
│   ├── wsi_inference.py        # WSI推理
│   ├── download_sam_checkpoints.py  # SAM下载
│   └── analyze_sweep.py        # 结果分析
├── run_hpo.py                  # HPO主程序
└── requirements.txt            # 依赖列表
```

---

## 🔧 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Zhaokun91/FTP.git
cd FTP
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 登录 W&B

```bash
wandb login
# 访问 wandb.ai 获取 API Key
```

---

## 🚀 快速开始

### 📦 数据准备（QuPath 用户）

如果您使用 QuPath 导出完整的 WSI 掩码，需要先切成小块：

```bash
# 1. 预览 patch 提取效果
python scripts/preview_patches.py \
    --image /path/to/wsi_image.png \
    --mask /path/to/wsi_mask.png \
    --patch-size 256

# 2. 提取 patches
python scripts/extract_patches.py \
    --image-dir /path/to/QuPath_Export \
    --mask-dir /path/to/QuPath_Export \
    --output-dir ./Patches \
    --patch-size 256 \
    --stride 256 \
    --min-foreground 0.05

# 3. 验证数据
python scripts/validate_data.py --data-dir ./Patches --show-sample
```

📚 **详细指南**: [数据准备完整文档](docs/DATA_PREPARATION.md)

---

## 🔮 推理和部署

训练完成后，使用最佳模型对新的 WSI 进行预测：

### 基础推理

```bash
# 对新的 WSI 进行预测
python scripts/wsi_inference.py \
    --image /path/to/new_wsi.png \
    --model /path/to/best_model.pth \
    --output ./predictions
```

### 高精度推理（推荐生产环境）

```bash
# 使用 TTA（测试时增强）+ 后处理 + QuPath 导出
python scripts/wsi_inference.py \
    --image /path/to/new_wsi.png \
    --model /path/to/best_model.pth \
    --output ./predictions \
    --tta \
    --post-process \
    --save-for-qupath \
    --stride 128
```

### 主要功能

- **滑动窗口推理**: 自动处理任意大小的 WSI
- **测试时增强 (TTA)**: 提升 2-5% Dice 分数（根据文献）
- **后处理**: 形态学操作，去除噪点和填充孔洞
- **QuPath 集成**: 直接导出为 QuPath 可导入格式
- **批量处理**: 一次处理多个 WSI

📚 **完整指南**: [推理和部署文档](docs/INFERENCE_GUIDE.md)

---

### 方法 1: Google Colab（推荐）

1. 打开 Colab Notebook: `notebooks/Osteoblast_Segmentation_HPO.ipynb`
2. 按照说明操作
3. 享受自动化优化！

### 方法 2: 本地运行

```bash
# 准备数据结构
# data/
#   ├── images/
#   └── labels/

# 运行 HPO
python run_hpo.py --config config/hpo_config.yaml --count 20
```

---

## 📖 详细使用指南

### 1. 配置搜索空间

编辑 `config/hpo_config.yaml`:

```yaml
parameters:
  model_name:
    values: ["U-Net", "U-Net++"]
  learning_rate:
    min: 1e-5
    max: 1e-3
  batch_size:
    values: [8, 16]
```

### 2. 启动 HPO

```bash
python run_hpo.py --config config/hpo_config.yaml --count 50
```

### 3. 监控进度

访问 W&B 仪表板查看实时进度。

### 4. 分析结果

```bash
python scripts/analyze_sweep.py --sweep_id YOUR_ID --project Osteoblast_HPO
```

---

## ⚙️ 配置说明

### 搜索方法

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| `bayes` | 贝叶斯优化 | **推荐**，效率最高 |
| `random` | 随机搜索 | 快速探索 |
| `grid` | 网格搜索 | 参数较少时 |

---

## 📊 结果分析

W&B 仪表板提供：
- 📈 平行坐标图 - 参数关系可视化
- 🎯 参数重要性分析
- 📉 性能对比图表
- 📊 详细结果表格

---

## ❓ 常见问题

**Q: HPO 需要多长时间？**
A: 约 `count × num_epochs × 每轮时间`。建议先用小参数测试。

**Q: 内存不足？**
A: 减小 `batch_size`、`image_size` 或使用更轻量的编码器。

**Q: Colab 超时？**
A: 使用 Colab Pro 或本地运行，可以继续未完成的 Sweep。

**Q: 如何使用 SAM/MedSAM？**
A: 查看详细的 [SAM 使用指南](docs/SAM_USAGE.md)，包含下载、配置和使用方法。

---

## 🎯 SAM/MedSAM 支持

系统现已支持 **Segment Anything Model (SAM)** 和 **MedSAM**！

### 快速开始使用 SAM

```bash
# 1. 下载 SAM 检查点
python scripts/download_sam_checkpoints.py --model vit_b --output ./checkpoints

# 2. 运行 SAM HPO
python run_hpo.py --config config/sam_hpo_config.yaml --count 10

# 3. SAM vs U-Net 对比
python run_hpo.py --config config/sam_vs_unet_config.yaml
```

### SAM 模型选项

| 模型 | 参数量 | 内存占用 | 推荐场景 |
|------|--------|---------|---------|
| SAM-ViT-B | 90M | ~4GB | 快速测试，资源受限 |
| SAM-ViT-L | 300M | ~8GB | 平衡性能 |
| SAM-ViT-H | 630M | ~16GB | 最高精度 |
| MedSAM | 90M | ~4GB | 医学图像（推荐） |

📚 **详细文档**: [SAM 完整使用指南](docs/SAM_USAGE.md)

---

## 🙏 致谢

- [Weights & Biases](https://wandb.ai) - 实验追踪
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [MONAI](https://monai.io/)

---

**如果这个项目对您有帮助，请给个 ⭐ Star！**