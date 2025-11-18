# 🔮 推理和部署指南

# Inference and Deployment Guide

将训练好的模型应用到新的 WSI 图像

---

## 📋 目录

- [快速开始](#快速开始)
- [WSI 推理](#wsi-推理)
- [测试时增强 (TTA)](#测试时增强-tta)
- [后处理](#后处理)
- [QuPath 集成](#qupath-集成)
- [性能优化](#性能优化)

---

## 🚀 快速开始

### 基础推理

```bash
# 对新的 WSI 进行预测
python scripts/wsi_inference.py \
    --image /path/to/new_wsi.png \
    --model /path/to/best_model.pth \
    --output ./predictions
```

### 高精度推理（推荐）

```bash
# 使用 TTA + 后处理 + QuPath 导出
python scripts/wsi_inference.py \
    --image /path/to/new_wsi.png \
    --model /path/to/best_model.pth \
    --output ./predictions \
    --tta \
    --post-process \
    --save-for-qupath \
    --stride 128
```

---

## 🎯 WSI 推理

### 工作原理

基于 **WSInfer** (Nature npj Precision Oncology 2024) 的标准流程：

```
完整 WSI 图像
    ↓
滑动窗口提取 patches
    ↓
批量预测每个 patch
    ↓
合并重叠区域（平均策略）
    ↓
生成完整预测掩码
```

### 参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `--patch-size` | Patch 大小 | `256` | 必须与训练时相同 |
| `--stride` | 滑动步长 | `128` (50%重叠) | 步长越小精度越高，速度越慢 |
| `--batch-size` | 批次大小 | `8-16` | 取决于 GPU 内存 |

### 重叠策略

**为什么需要重叠？**
- Patch 边界的预测质量较差
- 重叠可以平滑边界，提高整体精度

**重叠率对比：**

| Stride | 重叠率 | 精度 | 速度 | 推荐场景 |
|--------|--------|------|------|---------|
| `256` | 0% | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 快速预览 |
| `128` | 50% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **推荐，生产环境** |
| `64` | 75% | ⭐⭐⭐⭐⭐ | ⭐ | 最高精度要求 |

---

## 🔄 测试时增强 (TTA)

### 什么是 TTA？

**Test-Time Augmentation** - 在预测时对同一张图像应用多种变换，然后平均结果。

**原理：**
```python
原始图像 → 预测 → 结果1
水平翻转 → 预测 → 结果2 → 反向翻转
垂直翻转 → 预测 → 结果3 → 反向翻转
转置    → 预测 → 结果4 → 反向转置

最终结果 = 平均(结果1, 结果2, 结果3, 结果4)
```

### 使用 TTA

```bash
# 启用 TTA
python scripts/wsi_inference.py \
    --image wsi.png \
    --model best_model.pth \
    --output predictions \
    --tta  # 添加这个参数
```

### TTA 效果

根据文献 (Cell Segmentation 2024)：
- **Dice 提升**: +2-5%
- **边界质量**: 显著改善
- **代价**: 速度慢 2-4 倍

**推荐场景：**
- ✅ 最终生产环境
- ✅ 发表论文的结果
- ✅ 需要最高精度
- ❌ 快速预览
- ❌ 大批量处理

---

## 🔧 后处理

### 为什么需要后处理？

深度学习模型的原始输出可能包含：
- 小噪点
- 小孔洞
- 不规则边界

### 后处理流程

```python
原始预测
    ↓
形态学闭运算 → 填充小孔洞
    ↓
形态学开运算 → 去除小噪点
    ↓
连通组件分析 → 过滤小目标
    ↓
最终结果
```

### 使用后处理

```bash
python scripts/wsi_inference.py \
    --image wsi.png \
    --model best_model.pth \
    --output predictions \
    --post-process \          # 启用后处理
    --min-size 50             # 最小目标尺寸（像素）
```

### 参数调整

**`--min-size` 设置建议：**

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `10-30` | 保留几乎所有目标 | 骨细胞非常小 |
| `50` | 过滤小噪点 | **推荐** |
| `100+` | 只保留大目标 | 过滤假阳性 |

---

## 🔗 QuPath 集成

### 导出为 QuPath 格式

```bash
python scripts/wsi_inference.py \
    --image wsi.png \
    --model best_model.pth \
    --output predictions \
    --save-for-qupath  # 生成 QuPath 可导入的文件
```

### 在 QuPath 中导入

**步骤：**

1. **在 QuPath 中打开原始 WSI**
   ```
   File → Open → 选择您的 WSI 文件
   ```

2. **导入预测掩码为标注**
   ```
   Objects → Annotations → Import as annotations
   ```

3. **选择生成的掩码文件**
   ```
   选择: wsi_name_qupath.png
   ```

4. **设置参数**
   ```
   Threshold: 127
   Min area: 0 (或根据需要设置)
   Create objects for: Objects
   ```

5. **点击 OK**

预测结果会作为标注（Annotations）导入，您可以：
- 手动修正
- 导出统计数据
- 进一步分析

### QuPath 脚本自动化

在 QuPath 中运行以下 Groovy 脚本：

```groovy
// 自动导入预测掩码
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs

def maskPath = "/path/to/wsi_name_qupath.png"
def maskServer = ImageServerProvider.buildServer(maskPath)
def maskImage = maskServer.readBufferedImage()

// 转换为标注对象
// ... (QuPath API 调用)

println "预测掩码已导入！"
```

---

## ⚡ 性能优化

### GPU 优化

```bash
# 增大批次大小（如果 GPU 内存充足）
python scripts/wsi_inference.py \
    --image wsi.png \
    --model best_model.pth \
    --output predictions \
    --batch-size 16  # 从 8 增加到 16
```

### 内存优化

```bash
# 减小批次大小（如果内存不足）
python scripts/wsi_inference.py \
    --image wsi.png \
    --model best_model.pth \
    --output predictions \
    --batch-size 4  # 减小批次
```

### 速度 vs 精度权衡

| 配置 | 速度 | 精度 | 适用场景 |
|------|------|------|---------|
| `stride=256, no TTA` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 快速预览 |
| `stride=128, no TTA` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 日常使用 |
| `stride=128, TTA` | ⭐⭐ | ⭐⭐⭐⭐⭐ | **推荐，最终结果** |
| `stride=64, TTA` | ⭐ | ⭐⭐⭐⭐⭐ | 最高精度 |

---

## 📊 批量推理

### 处理多个 WSI

创建批处理脚本：

```bash
#!/bin/bash
# batch_inference.sh

WSI_DIR="/path/to/wsi_images"
MODEL="/path/to/best_model.pth"
OUTPUT_DIR="/path/to/predictions"

for wsi in $WSI_DIR/*.png; do
    echo "处理: $wsi"
    python scripts/wsi_inference.py \
        --image "$wsi" \
        --model "$MODEL" \
        --output "$OUTPUT_DIR/$(basename $wsi .png)" \
        --tta \
        --post-process \
        --save-for-qupath
done

echo "所有 WSI 处理完成！"
```

使用：
```bash
chmod +x batch_inference.sh
./batch_inference.sh
```

---

## 🎓 完整示例

### 场景：最终发表论文的结果

```bash
# 步骤 1: 找到最佳模型
# 从 W&B 仪表板找到 Dice 最高的模型
# 假设是: ./models/run_42_best_model.pth

# 步骤 2: 高精度推理
python scripts/wsi_inference.py \
    --image ./test_wsi/slide_test_001.png \
    --model ./models/run_42_best_model.pth \
    --output ./final_predictions/slide_001 \
    --patch-size 256 \
    --stride 128 \
    --batch-size 8 \
    --tta \
    --post-process \
    --min-size 50 \
    --save-for-qupath

# 步骤 3: 在 QuPath 中验证
# - 打开 slide_test_001
# - 导入 slide_test_001_qupath.png
# - 手动检查和修正
# - 导出最终标注

# 步骤 4: 计算评估指标
python scripts/evaluate_predictions.py \
    --pred ./final_predictions/slide_001/slide_test_001_prediction.png \
    --gt ./ground_truth/slide_test_001_mask.png \
    --output ./metrics.csv
```

---

## ❓ 常见问题

### Q1: 推理很慢怎么办？

**A:**
1. 减小 `stride`（牺牲少量精度）
2. 关闭 TTA
3. 增大 `batch-size`（如果 GPU 内存充足）
4. 使用更小的模型（如 U-Net 而非 U-Net++）

### Q2: 内存不足 (OOM)

**A:**
```bash
# 减小批次大小
--batch-size 2

# 或使用 CPU（很慢）
export CUDA_VISIBLE_DEVICES=""
python scripts/wsi_inference.py ...
```

### Q3: 预测结果有很多噪点

**A:**
```bash
# 启用后处理，增大最小尺寸
--post-process --min-size 100
```

### Q4: QuPath 导入后找不到标注

**A:**
检查：
1. 掩码阈值设置：使用 127
2. 确保原始 WSI 和预测掩码尺寸一致
3. 检查预测掩码不是全黑（没有检测到任何骨细胞）

### Q5: 如何评估预测质量？

**A:**
```python
# 计算 Dice 分数
from sklearn.metrics import f1_score

pred = cv2.imread('prediction.png', 0) > 127
gt = cv2.imread('ground_truth.png', 0) > 127

dice = f1_score(gt.flatten(), pred.flatten())
print(f"Dice: {dice:.4f}")
```

---

## 📚 参考文献

1. **WSInfer**: Kaczmarzyk, J.R., et al. (2024). "Open and reusable deep learning for pathology with WSInfer and QuPath." npj Precision Oncology, 8, 9.

2. **TTA for Cell Segmentation**: Al-Kofahi, Y., et al. (2024). "Deep learning for cell segmentation." Neural Computing and Applications.

3. **QuPath**: Bankhead, P., et al. (2017). "QuPath: Open source software for digital pathology image analysis." Scientific Reports, 7, 16878.

---

**下一步**: 查看 [主 README](../README.md) 或 [训练指南](../README.md#详细使用指南)
