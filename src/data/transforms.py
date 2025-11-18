"""
数据增强和预处理
Data Augmentation and Preprocessing
"""

from typing import Optional
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation(
    image_size: tuple = (256, 256),
    augmentation_prob: float = 0.5
) -> albu.Compose:
    """
    训练集数据增强

    Args:
        image_size: 目标图像大小 (height, width)
        augmentation_prob: 增强概率

    Returns:
        Albumentations Compose对象
    """
    train_transform = [
        # 调整大小
        albu.Resize(height=image_size[0], width=image_size[1], always_apply=True),

        # 几何变换
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=augmentation_prob
        ),

        # 弹性变形（模拟组织变形）
        albu.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            border_mode=cv2.BORDER_CONSTANT,
            p=augmentation_prob * 0.5
        ),

        # 颜色增强
        albu.OneOf([
            albu.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            albu.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            albu.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=augmentation_prob),

        # 模糊和噪声（模拟显微镜图像质量变化）
        albu.OneOf([
            albu.GaussianBlur(blur_limit=(3, 7), p=1.0),
            albu.MedianBlur(blur_limit=5, p=1.0),
            albu.MotionBlur(blur_limit=5, p=1.0),
        ], p=augmentation_prob * 0.3),

        albu.GaussNoise(var_limit=(10.0, 50.0), p=augmentation_prob * 0.3),

        # 归一化到 [0, 1]
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet统计值
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        # 转换为PyTorch张量
        ToTensorV2(),
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation(
    image_size: tuple = (256, 256)
) -> albu.Compose:
    """
    验证集数据增强（仅调整大小和归一化）

    Args:
        image_size: 目标图像大小 (height, width)

    Returns:
        Albumentations Compose对象
    """
    val_transform = [
        albu.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]

    return albu.Compose(val_transform)


def get_preprocessing(preprocessing_fn: Optional[callable] = None) -> albu.Compose:
    """
    获取预处理函数

    Args:
        preprocessing_fn: 自定义预处理函数（如编码器的预处理）

    Returns:
        Albumentations Compose对象
    """
    _transform = []

    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))

    _transform.append(ToTensorV2())

    return albu.Compose(_transform)
