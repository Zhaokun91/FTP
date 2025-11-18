# 📦 数据准备指南

# Data Preparation Guide

从 QuPath 导出的完整 WSI 掩码到训练用的 patches

---

## 📋 目录

- [工作流程](#工作流程)
- [QuPath 导出](#qupath-导出)
- [Patch 提取](#patch-提取)
- [数据验证](#数据验证)
- [常见问题](#常见问题)

---

## 🔄 工作流程

```
QuPath 标注
    ↓
导出 WSI 图像和掩码
    ↓
预览 Patch 提取效果（可选）
    ↓
提取 Patches
    ↓
验证数据格式
    ↓
开始训练
```

---

## 🖼️ QuPath 导出

### 1. 在 QuPath 中标注骨细胞

1. 打开 WSI 图像
2. 使用标注工具标记骨细胞区域
3. 分类设置：
   - 背景：不标注或标注为"Background"
   - 骨细胞：标注为"Osteoblast"或其他类别

### 2. 导出图像和掩码

**方法 A: 使用 QuPath 脚本**

```groovy
// QuPath 导出脚本
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.regions.RegionRequest

// 设置导出参数
def imageData = getCurrentImageData()
def server = imageData.getServer()
def downsample = 1.0  // 下采样因子（1.0 = 原始分辨率）

// 导出完整图像
def outputDir = "C:/QuPath_Export"  // 修改为您的路径
def imageName = server.getMetadata().getName()

// 导出原图
def requestImage = RegionRequest.createInstance(server, downsample)
def img = server.readBufferedImage(requestImage)
def imageFile = new File(outputDir, imageName + "_image.png")
javax.imageio.ImageIO.write(img, "PNG", imageFile)

// 导出掩码
def requestMask = RegionRequest.createInstance(server, downsample)
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK)  // 背景 = 0
    .addLabel('Osteoblast', 1)             // 骨细胞 = 1
    .downsample(downsample)
    .build()

def mask = labelServer.readBufferedImage(requestMask)
def maskFile = new File(outputDir, imageName + "_mask.png")
javax.imageio.ImageIO.write(mask, "PNG", maskFile)

print "导出完成！"
```

**方法 B: 手动导出**

1. 文件 → Export images → Rendered RGB (或 Original pixels)
2. 文件 → Export images → Labelled image
   - 设置背景为 0
   - 设置骨细胞为 1 或 255

### 3. 预期的导出结果

```
QuPath_Export/
├── slide_001_image.png    # 原始图像
├── slide_001_mask.png     # 对应掩码
├── slide_002_image.png
├── slide_002_mask.png
└── ...
```

**掩码格式要求：**
- 背景像素值：`0`
- 骨细胞像素值：`1` 或 `255`
- 图像和掩码尺寸必须完全一致

---

## ✂️ Patch 提取

### 1. 预览效果（推荐先做）

在正式提取之前，预览一下会生成多少 patches：

```bash
python scripts/preview_patches.py \
    --image /path/to/slide_001_image.png \
    --mask /path/to/slide_001_mask.png \
    --patch-size 256 \
    --stride 256 \
    --min-foreground 0.05
```

**输出：**
- 生成 `patch_extraction_preview.png`
- 显示 patch 分布、前景比例直方图
- 预估将提取的 patch 数量

### 2. 正式提取 Patches

```bash
python scripts/extract_patches.py \
    --image-dir /path/to/QuPath_Export \
    --mask-dir /path/to/QuPath_Export \
    --output-dir /path/to/Patches \
    --patch-size 256 \
    --stride 128 \
    --min-foreground 0.05
```

**参数说明：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--image-dir` | 包含 WSI 图像的目录 | QuPath 导出目录 |
| `--mask-dir` | 包含 WSI 掩码的目录 | 通常与 image-dir 相同 |
| `--output-dir` | 输出目录 | `./Patches` |
| `--patch-size` | Patch 大小 | `256` 或 `512` |
| `--stride` | 滑动步长 | `128`（50%重叠）或 `256`（无重叠） |
| `--min-foreground` | 最小前景比例 | `0.05`（5%） |
| `--background-value` | 背景值 | `0` |
| `--save-format` | 保存格式 | `png` |
| `--overwrite` | 覆盖已存在的输出 | - |

**示例：**

```bash
# 示例 1: 无重叠，只保留前景较多的 patch
python scripts/extract_patches.py \
    --image-dir ./QuPath_Export \
    --mask-dir ./QuPath_Export \
    --output-dir ./Patches \
    --patch-size 256 \
    --stride 256 \
    --min-foreground 0.1

# 示例 2: 50% 重叠，保留所有包含骨细胞的 patch
python scripts/extract_patches.py \
    --image-dir ./QuPath_Export \
    --mask-dir ./QuPath_Export \
    --output-dir ./Patches_Overlap \
    --patch-size 512 \
    --stride 256 \
    --min-foreground 0.01

# 示例 3: 使用自定义文件名规则
python scripts/extract_patches.py \
    --image-dir ./WSI_Images \
    --mask-dir ./WSI_Masks \
    --output-dir ./Training_Data \
    --patch-size 256 \
    --save-format tif
```

### 3. 输出结果

```
Patches/
├── images/
│   ├── slide_001_y00000_x00000_idx0000.png
│   ├── slide_001_y00000_x00256_idx0001.png
│   ├── slide_001_y00256_x00000_idx0002.png
│   └── ... (可能有数百到数千个文件)
└── labels/
    ├── slide_001_y00000_x00000_idx0000.png
    ├── slide_001_y00000_x00256_idx0001.png
    ├── slide_001_y00256_x00000_idx0002.png
    └── ...
```

**文件命名规则：**
- `{原图名}_y{起始Y坐标}_x{起始X坐标}_idx{索引}.png`
- 例如：`slide_001_y01024_x02048_idx0042.png`
  - 来自 `slide_001`
  - 在原图的 (x=2048, y=1024) 位置
  - 是第 42 个提取的 patch

---

## ✅ 数据验证

提取完成后，验证数据格式：

```bash
# 基础验证
python scripts/validate_data.py --data-dir ./Patches

# 详细验证 + 可视化样本
python scripts/validate_data.py \
    --data-dir ./Patches \
    --show-sample \
    --num-samples 20
```

**验证内容：**
- ✅ 图像和掩码数量是否匹配
- ✅ 掩码值是否正确（0 和 1 或 0 和 255）
- ✅ 文件是否可读
- ✅ 前景比例统计

**预期输出：**
```
✅ 格式正确！掩码使用 0 (背景) 和 1 (骨细胞)
   或
✅ 格式可接受！掩码使用 0 (背景) 和 255 (骨细胞)
   ℹ️  系统会自动转换为 0 和 1
```

---

## 🎯 参数调优建议

### Patch Size（Patch 大小）

| 大小 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 128x128 | 数据量大，训练快 | 上下文少 | 小目标，简单场景 |
| **256x256** | 平衡 ✅ | - | **推荐，通用** |
| 512x512 | 上下文丰富 | 内存占用大，数据量少 | 大目标，需要全局信息 |
| 1024x1024 | 最大上下文 | 训练慢，需要大显存 | SAM 模型 |

### Stride（滑动步长）

| 设置 | 重叠率 | 数据量 | 适用场景 |
|------|--------|--------|---------|
| `stride = patch_size` | 0% | 少 | 数据充足 |
| `stride = patch_size / 2` | 50% | 4倍 | **推荐，增强数据** |
| `stride = patch_size / 4` | 75% | 16倍 | 数据很少 |

### Min Foreground（最小前景比例）

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.01 | 保留几乎所有 patch | 骨细胞很稀疏 |
| **0.05** | 过滤大部分空白 | **推荐** |
| 0.10 | 只保留前景丰富的 | 骨细胞密集 |
| 0.20 | 非常严格 | 聚焦高密度区域 |

---

## ❓ 常见问题

### Q1: 提取了太多 patches，怎么办？

**A:** 调整参数减少数据量：
```bash
# 增大步长（减少重叠）
--stride 256  # 从 128 改为 256

# 提高前景阈值
--min-foreground 0.1  # 从 0.05 改为 0.1

# 增大 patch 大小
--patch-size 512  # 从 256 改为 512
```

### Q2: 提取的 patches 太少，怎么办？

**A:** 调整参数增加数据量：
```bash
# 减小步长（增加重叠）
--stride 128  # 50% 重叠

# 降低前景阈值
--min-foreground 0.01  # 保留更多 patch

# 减小 patch 大小
--patch-size 128
```

### Q3: 图像和掩码尺寸不匹配

**A:**
1. 检查 QuPath 导出设置，确保下采样因子（downsample）一致
2. 使用图像处理工具调整尺寸：
```python
import cv2
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
mask_resized = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('mask_resized.png', mask_resized)
```

### Q4: 掩码值不是 0 和 1/255

**A:** 使用脚本转换：
```python
import cv2
import numpy as np

mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
# 二值化
mask_binary = (mask > 127).astype(np.uint8) * 255
cv2.imwrite('mask_binary.png', mask_binary)
```

### Q5: 内存不足

**A:**
1. 一次处理一个 WSI
2. 减小 patch 大小
3. 使用生成器而非一次性加载所有数据

---

## 📚 完整工作流程示例

### 场景：从 QuPath 导出到开始训练

```bash
# 步骤 1: 在 QuPath 中标注并导出
# （手动操作，导出到 ./QuPath_Export）

# 步骤 2: 预览一个 WSI 的 patch 提取效果
python scripts/preview_patches.py \
    --image ./QuPath_Export/slide_001_image.png \
    --mask ./QuPath_Export/slide_001_mask.png \
    --patch-size 256 \
    --stride 256

# 查看 patch_extraction_preview.png，确认参数合适

# 步骤 3: 提取所有 WSI 的 patches
python scripts/extract_patches.py \
    --image-dir ./QuPath_Export \
    --mask-dir ./QuPath_Export \
    --output-dir ./Patches \
    --patch-size 256 \
    --stride 256 \
    --min-foreground 0.05

# 步骤 4: 验证数据
python scripts/validate_data.py \
    --data-dir ./Patches \
    --show-sample

# 步骤 5: 开始 HPO 训练
python run_hpo.py \
    --config config/simple_hpo_config.yaml \
    --count 10
```

---

## 🎓 进阶技巧

### 1. 分层采样

如果骨细胞分布不均，可以分别处理高密度和低密度区域：

```bash
# 高密度区域（严格筛选）
python scripts/extract_patches.py \
    --image-dir ./QuPath_Export \
    --mask-dir ./QuPath_Export \
    --output-dir ./Patches_HighDensity \
    --min-foreground 0.2

# 低密度区域（宽松筛选）
python scripts/extract_patches.py \
    --image-dir ./QuPath_Export \
    --mask-dir ./QuPath_Export \
    --output-dir ./Patches_LowDensity \
    --min-foreground 0.02

# 合并
cp -r ./Patches_HighDensity/images/* ./Patches_Final/images/
cp -r ./Patches_LowDensity/images/* ./Patches_Final/images/
# （掩码同理）
```

### 2. 数据增强

提取 patches 后，可以使用脚本进行离线增强：

```python
# augment_patches.py
import cv2
import albumentations as albu

transform = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomRotate90(p=0.5),
])

# 对每个 patch 应用增强...
```

### 3. 质量控制

手动检查提取的 patches：

```bash
# 随机抽样检查
python -c "
import random
from pathlib import Path
patches = list(Path('./Patches/images').glob('*.png'))
samples = random.sample(patches, 20)
for p in samples:
    print(p)
"
```

---

**下一步：** 查看 [主 README](../README.md) 开始训练！

