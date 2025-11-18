"""
SAM (Segment Anything Model) 适配器
SAM Adapter for Medical Image Segmentation

将 SAM/MedSAM 适配为传统分割模型接口，使其能够在 HPO 系统中使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import warnings


class SAMSegmentationAdapter(nn.Module):
    """
    SAM 分割适配器

    将 SAM 的提示式分割转换为端到端的自动分割，
    使其能够与 U-Net 等模型在同一框架下训练和比较。

    策略：
    1. 自动生成提示点（基于图像特征或网格采样）
    2. 使用 SAM 进行分割
    3. 支持微调 SAM 的解码器部分
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        num_classes: int = 2,
        prompt_strategy: str = "grid",
        num_prompts: int = 16
    ):
        """
        Args:
            model_type: SAM 模型类型 ("vit_b", "vit_l", "vit_h")
            checkpoint_path: 预训练模型路径
            freeze_encoder: 是否冻结编码器
            num_classes: 输出类别数
            prompt_strategy: 提示生成策略 ("grid", "center", "random")
            num_prompts: 提示点数量
        """
        super().__init__()

        try:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam_available = True
        except ImportError:
            warnings.warn("segment-anything not installed. SAM models will not be available.")
            self.sam_available = False
            return

        self.model_type = model_type
        self.num_classes = num_classes
        self.prompt_strategy = prompt_strategy
        self.num_prompts = num_prompts

        # 加载 SAM 模型
        if checkpoint_path is None:
            # 使用默认路径或提示用户下载
            checkpoint_path = self._get_default_checkpoint(model_type)

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        # 是否冻结编码器
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        # 添加分类头（将 SAM 的输出转换为多类分割）
        if num_classes > 2:
            self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            self.classifier = None

    def _get_default_checkpoint(self, model_type: str) -> str:
        """获取默认的检查点路径"""
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

        # 提示用户下载
        url = checkpoint_urls.get(model_type)
        raise ValueError(
            f"Please download SAM checkpoint from {url} "
            f"and provide the path via checkpoint_path parameter."
        )

    def generate_prompts(
        self,
        image: torch.Tensor,
        strategy: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成提示点

        Args:
            image: 输入图像 (B, C, H, W)
            strategy: 提示策略

        Returns:
            point_coords: 提示点坐标 (B, N, 2)
            point_labels: 提示点标签 (B, N) - 1表示前景，0表示背景
        """
        if strategy is None:
            strategy = self.prompt_strategy

        B, C, H, W = image.shape
        device = image.device

        if strategy == "grid":
            # 网格采样
            grid_size = int(np.sqrt(self.num_prompts))
            x = torch.linspace(0, W-1, grid_size, device=device)
            y = torch.linspace(0, H-1, grid_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            points = points.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 2)
            labels = torch.ones(B, points.shape[1], device=device)  # 全部为前景点

        elif strategy == "center":
            # 中心点
            center_x = W // 2
            center_y = H // 2
            points = torch.tensor([[center_x, center_y]], device=device).float()
            points = points.unsqueeze(0).repeat(B, 1, 1)
            labels = torch.ones(B, 1, device=device)

        elif strategy == "random":
            # 随机采样
            points = torch.rand(B, self.num_prompts, 2, device=device)
            points[:, :, 0] *= W
            points[:, :, 1] *= H
            labels = torch.ones(B, self.num_prompts, device=device)

        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}")

        return points, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            output: 分割结果 (B, num_classes, H, W)
        """
        if not self.sam_available:
            raise RuntimeError("SAM is not available. Please install segment-anything.")

        B, C, H, W = x.shape
        device = x.device

        # 1. 图像编码
        image_embeddings = self.sam.image_encoder(x)

        # 2. 生成提示
        point_coords, point_labels = self.generate_prompts(x)

        # 3. 提示编码和掩码解码
        outputs = []
        for i in range(B):
            # SAM 的提示编码器
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point_coords[i:i+1], point_labels[i:i+1]),
                boxes=None,
                masks=None
            )

            # SAM 的掩码解码器
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            outputs.append(low_res_masks)

        # 4. 上采样到原始分辨率
        masks = torch.cat(outputs, dim=0)  # (B, 1, H', W')
        masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)

        # 5. 转换为多类输出
        if self.num_classes == 2:
            # 二分类：背景 + 前景
            background = 1 - masks
            output = torch.cat([background, masks], dim=1)  # (B, 2, H, W)
        else:
            # 多分类：使用额外的分类头
            if self.classifier is not None:
                output = self.classifier(masks)
            else:
                raise ValueError("Classifier not initialized for multi-class segmentation")

        return output


class MedSAMAdapter(nn.Module):
    """
    MedSAM 适配器

    MedSAM 是专门为医学图像优化的 SAM 变体
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False,  # MedSAM 通常需要微调
        num_classes: int = 2,
        image_size: int = 256
    ):
        """
        Args:
            checkpoint_path: MedSAM 预训练模型路径
            freeze_encoder: 是否冻结编码器
            num_classes: 输出类别数
            image_size: 输入图像大小
        """
        super().__init__()

        try:
            from medsam import build_medsam
            self.medsam_available = True
        except ImportError:
            warnings.warn("MedSAM not installed. Install it via: pip install git+https://github.com/bowang-lab/MedSAM.git")
            self.medsam_available = False
            return

        self.num_classes = num_classes
        self.image_size = image_size

        # 加载 MedSAM
        self.medsam = build_medsam()

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.medsam.load_state_dict(checkpoint, strict=False)

        # 冻结编码器
        if freeze_encoder:
            for param in self.medsam.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            output: 分割结果 (B, num_classes, H, W)
        """
        if not self.medsam_available:
            raise RuntimeError("MedSAM is not available.")

        B, C, H, W = x.shape

        # MedSAM 的前向传播
        # 注意：MedSAM 可能需要特定的输入处理
        output = self.medsam(x)

        # 调整输出格式
        if output.shape[1] == 1:
            # 单通道输出转为双通道
            background = 1 - torch.sigmoid(output)
            foreground = torch.sigmoid(output)
            output = torch.cat([background, foreground], dim=1)

        return output


def create_sam_model(
    model_name: str = "SAM-ViT-B",
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    创建 SAM 模型

    Args:
        model_name: 模型名称
            - "SAM-ViT-B": SAM with ViT-B encoder
            - "SAM-ViT-L": SAM with ViT-L encoder
            - "SAM-ViT-H": SAM with ViT-H encoder
            - "MedSAM": Medical SAM
        checkpoint_path: 模型检查点路径
        **kwargs: 其他参数

    Returns:
        SAM 模型
    """
    model_name = model_name.upper()

    if "MEDSAM" in model_name:
        return MedSAMAdapter(checkpoint_path=checkpoint_path, **kwargs)

    elif "SAM" in model_name:
        # 提取 ViT 类型
        if "VIT-B" in model_name or "VIT_B" in model_name:
            model_type = "vit_b"
        elif "VIT-L" in model_name or "VIT_L" in model_name:
            model_type = "vit_l"
        elif "VIT-H" in model_name or "VIT_H" in model_name:
            model_type = "vit_h"
        else:
            model_type = "vit_b"  # 默认

        return SAMSegmentationAdapter(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown SAM model: {model_name}")
