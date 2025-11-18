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
- U-Net, U-Net++, FPN, DeepLabV3+, PSPNet, LinkNet

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
│   └── simple_hpo_config.yaml  # 简化版配置
├── src/                         # 源代码
│   ├── data/                   # 数据处理
│   ├── models/                 # 模型定义
│   ├── utils/                  # 工具函数
│   └── train.py                # 训练逻辑
├── notebooks/                   # Jupyter Notebooks
│   └── Osteoblast_Segmentation_HPO.ipynb
├── scripts/                     # 辅助脚本
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

---

## 🙏 致谢

- [Weights & Biases](https://wandb.ai) - 实验追踪
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [MONAI](https://monai.io/)

---

**如果这个项目对您有帮助，请给个 ⭐ Star！**