# 🎯 SAM/MedSAM 使用指南

# SAM (Segment Anything Model) Usage Guide

---

## 📋 目录

- [简介](#简介)
- [安装](#安装)
- [下载预训练模型](#下载预训练模型)
- [快速开始](#快速开始)
- [HPO 配置](#hpo-配置)
- [SAM vs U-Net 对比](#sam-vs-u-net-对比)
- [常见问题](#常见问题)
- [性能对比](#性能对比)

---

## 🎯 简介

### 什么是 SAM？

**SAM (Segment Anything Model)** 是 Meta AI 开发的通用分割模型：
- 基于 Vision Transformer (ViT) 架构
- 使用提示（prompts）进行分割
- 在 11M 图像上预训练

### 什么是 MedSAM？

**MedSAM** 是专门为医学图像优化的 SAM 变体：
- 在大规模医学图像数据集上微调
- 更适合医学图像的特征
- 对细胞、组织等医学对象有更好的分割性能

### 为什么要用 SAM？

| 优势 | 说明 |
|------|------|
| **强大的预训练** | 在海量数据上预训练，泛化能力强 |
| **Zero-shot 能力** | 无需训练即可使用 |
| **高质量分割** | 边界精细，细节丰富 |
| **医学图像优化** | MedSAM 专门针对医学图像 |

---

## 🔧 安装

### 1. 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 或者单独安装 SAM 相关包
pip install segment-anything
pip install timm
pip install git+https://github.com/bowang-lab/MedSAM.git
```

### 2. 验证安装

```python
import segment_anything
import torch

print(f"SAM version: {segment_anything.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📥 下载预训练模型

### 自动下载（推荐）

使用我们提供的下载脚本：

```bash
# 下载 SAM-ViT-B（轻量级，推荐）
python scripts/download_sam_checkpoints.py --model vit_b --output ./checkpoints

# 下载 SAM-ViT-L（平衡）
python scripts/download_sam_checkpoints.py --model vit_l --output ./checkpoints

# 下载 SAM-ViT-H（最强性能）
python scripts/download_sam_checkpoints.py --model vit_h --output ./checkpoints

# 下载 MedSAM（医学图像专用）
python scripts/download_sam_checkpoints.py --model medsam --output ./checkpoints

# 下载所有模型
python scripts/download_sam_checkpoints.py --model all --output ./checkpoints
```

### 手动下载

**SAM 官方模型：**
- ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

**MedSAM：**
- https://zenodo.org/records/10155347/files/medsam_vit_b.pth

下载后放到 `checkpoints/` 目录。

---

## 🚀 快速开始

### 方法 1: 使用 HPO 系统

```bash
# 1. 下载 SAM 检查点
python scripts/download_sam_checkpoints.py --model vit_b

# 2. 配置检查点路径（编辑 config/sam_hpo_config.yaml）
# checkpoint_path: "./checkpoints/sam_vit_b_01ec64.pth"

# 3. 运行 HPO
python run_hpo.py --config config/sam_hpo_config.yaml --count 10
```

### 方法 2: 直接训练

```python
from src.models.model_factory import create_model

# 创建 SAM 模型
model = create_model(
    model_name="SAM-ViT-B",
    checkpoint_path="./checkpoints/sam_vit_b_01ec64.pth",
    freeze_encoder=True,  # 仅微调解码器
    num_classes=2
)

# 正常训练...
```

### 方法 3: Google Colab

在 Colab Notebook 中：

```python
# Cell 1: 下载 SAM 检查点
!python scripts/download_sam_checkpoints.py --model vit_b --output /content/checkpoints

# Cell 2: 配置 HPO
sweep_config = {
    'parameters': {
        'model_name': {'values': ['SAM-ViT-B', 'MedSAM']},
        'freeze_encoder': {'values': [True, False]},
        # ... 其他参数
    }
}

# Cell 3: 运行
wandb.agent(sweep_id, function=train_one_run, count=10)
```

---

## ⚙️ HPO 配置

### SAM 专用配置

编辑 `config/sam_hpo_config.yaml`：

```yaml
parameters:
  # 模型选择
  model_name:
    values:
      - "SAM-ViT-B"      # 推荐：轻量快速
      - "SAM-ViT-L"      # 平衡性能
      - "SAM-ViT-H"      # 最高精度（需要大显存）
      - "MedSAM"         # 医学图像专用

  # SAM 特定参数
  freeze_encoder: {values: [true, false]}
  prompt_strategy: {values: ["grid", "center"]}
  num_prompts: {values: [9, 16, 25]}

  # 训练参数（SAM 特殊配置）
  learning_rate: {min: 1e-6, max: 1e-4}  # 较小学习率
  batch_size: {values: [1, 2, 4]}         # 较小批次
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `freeze_encoder` | 冻结编码器，仅训练解码器 | `true`（数据少时） |
| `prompt_strategy` | 提示点生成策略 | `"grid"`（均匀采样） |
| `num_prompts` | 提示点数量 | `16`（4x4 网格） |
| `learning_rate` | 学习率 | `1e-5`（较小） |
| `batch_size` | 批次大小 | `2`（SAM 内存占用大） |
| `image_size` | 图像大小 | `[1024, 1024]`（SAM原生） |

---

## 📊 SAM vs U-Net 对比

### 运行对比实验

```bash
# 使用对比配置
python run_hpo.py --config config/sam_vs_unet_config.yaml --count 30
```

这会同时测试：
- U-Net + ResNet34
- U-Net++ + EfficientNet-B0
- SAM-ViT-B
- MedSAM

### 预期结果

基于文献报告和经验：

| 模型 | Dice | 训练时间 | 推理速度 | 内存占用 |
|------|------|---------|---------|---------|
| **U-Net** | 0.85-0.90 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **U-Net++** | 0.87-0.92 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **SAM-ViT-B** | 0.82-0.88 | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **MedSAM** | 0.85-0.90 | ⭐⭐ | ⭐⭐ | ⭐⭐ |

**结论：**
- **U-Net 系列**：训练快，推理快，适合生产环境
- **SAM/MedSAM**：预训练强大，适合数据少的情况

---

## ❓ 常见问题

### Q1: SAM 下载失败怎么办？

**A:**
1. 检查网络连接
2. 使用代理或镜像
3. 手动下载后放到 `checkpoints/` 目录

```bash
# 使用 wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth checkpoints/
```

### Q2: 内存不足 (CUDA OOM)

**A:**
- 减小 `batch_size` 到 1 或 2
- 使用 ViT-B 而非 ViT-H
- 减小 `image_size` 到 512x512
- 启用梯度检查点：

```python
model = create_model(
    model_name="SAM-ViT-B",
    use_gradient_checkpointing=True  # 牺牲速度换内存
)
```

### Q3: SAM 训练很慢

**A:**
- 正常现象，SAM 参数量大（ViT-B: 90M, ViT-H: 630M）
- 建议：
  - 冻结编码器 (`freeze_encoder: true`)
  - 减少实验次数 (`count: 10` 而非 50)
  - 使用更少的 epochs (`num_epochs: 20`)

### Q4: MedSAM vs SAM 如何选择？

**A:**

| 场景 | 推荐 |
|------|------|
| 医学图像（细胞、组织） | **MedSAM** |
| 自然图像 | SAM |
| 数据量大 | SAM（从头训练） |
| 数据量小 | MedSAM（医学预训练） |

### Q5: 如何解读提示策略？

**A:**

```python
# grid: 均匀网格采样
# 适合：完整覆盖整个图像，适合分散的小目标
# num_prompts=16 → 4x4 网格

# center: 中心点
# 适合：目标位于图像中心
# num_prompts=1

# random: 随机采样
# 适合：探索不同位置的影响
```

---

## 🎓 进阶用法

### 1. 自定义提示策略

编辑 `src/models/sam_adapter.py`：

```python
def generate_prompts_custom(self, image, mask_hint=None):
    """
    基于掩码提示生成点

    Args:
        mask_hint: 粗略的掩码提示
    """
    # 从掩码中心采样点
    # ...
```

### 2. 混合 SAM 和 U-Net

```python
# 使用 SAM 生成伪标签，再训练 U-Net
sam_predictions = sam_model(images)
unet_model.train(images, sam_predictions)
```

### 3. 层次化微调

```python
# 第1阶段：冻结编码器
model.freeze_encoder = True
train(model, epochs=20)

# 第2阶段：解冻编码器
model.freeze_encoder = False
train(model, epochs=10, lr=1e-6)
```

---

## 📚 参考资源

- [SAM 官方论文](https://arxiv.org/abs/2304.02643)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [MedSAM 论文](https://arxiv.org/abs/2304.12306)
- [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)
- [W&B SAM 教程](https://wandb.ai/geekyrakshit/segment-anything)

---

**祝您使用 SAM 愉快！** 🎉

如有问题，请查看 [主 README](../README.md) 或提交 Issue。
