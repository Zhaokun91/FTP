"""
模型工厂
Model Factory for creating segmentation models
"""

import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Optional


def create_model(
    model_name: str,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
    activation: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    创建分割模型

    Args:
        model_name: 模型名称 (U-Net, U-Net++, FPN, DeepLabV3+等)
        encoder_name: 编码器骨干网络名称
        encoder_weights: 预训练权重 ("imagenet" 或 None)
        in_channels: 输入通道数
        num_classes: 输出类别数
        activation: 激活函数 (None, "sigmoid", "softmax")
        **kwargs: 其他模型参数

    Returns:
        PyTorch模型
    """
    model_name = model_name.lower().replace("-", "").replace("_", "")

    # 支持的模型字典
    model_dict = {
        "unet": smp.Unet,
        "unet++": smp.UnetPlusPlus,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3+": smp.DeepLabV3Plus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "pan": smp.PAN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
    }

    if model_name not in model_dict:
        available_models = ", ".join(model_dict.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available_models}"
        )

    # 创建模型
    model_class = model_dict[model_name]

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
        **kwargs
    )

    return model


def get_model_info(model: nn.Module) -> dict:
    """
    获取模型信息

    Args:
        model: PyTorch模型

    Returns:
        包含模型参数量等信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }

    return info
