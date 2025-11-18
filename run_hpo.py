#!/usr/bin/env python3
"""
超参数优化主脚本
Main Script for Hyperparameter Optimization with W&B Sweeps

使用方法:
    python run_hpo.py --config config/hpo_config.yaml --count 50

作者: FTP Project
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import wandb
import yaml

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import prepare_dataloaders
from src.data.transforms import get_training_augmentation, get_validation_augmentation
from src.models.model_factory import create_model
from src.models.losses import get_loss_function
from src.train import train_model
from src.utils.utils import set_seed, get_device


def train_one_run(config=None):
    """
    单次训练运行（被W&B Sweep调用）

    这个函数会被W&B Sweep agent自动调用多次，
    每次使用不同的超参数组合

    Args:
        config: W&B配置对象（由sweep agent自动传入）
    """
    # 初始化W&B run
    with wandb.init(config=config) as run:
        # 获取配置
        config = wandb.config

        print("\n" + "="*70)
        print(f"Starting Run: {run.name}")
        print("="*70)
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("="*70 + "\n")

        # 设置随机种子
        set_seed(config.random_seed)

        # 获取设备
        device = get_device()

        # 准备数据增强
        train_transform = get_training_augmentation(
            image_size=tuple(config.image_size),
            augmentation_prob=config.get('augmentation_prob', 0.5)
        )
        val_transform = get_validation_augmentation(
            image_size=tuple(config.image_size)
        )

        # 准备数据加载器
        print("Preparing data loaders...")
        try:
            train_loader, val_loader = prepare_dataloaders(
                data_dir=config.data_dir,
                batch_size=config.batch_size,
                val_split=config.val_split,
                num_workers=config.num_workers,
                random_seed=config.random_seed,
                train_transform=train_transform,
                val_transform=val_transform
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please make sure the data directory exists and contains 'images' and 'labels' folders")
            return

        # 创建模型
        print(f"\nCreating model: {config.model_name} with encoder: {config.encoder_name}...")
        model = create_model(
            model_name=config.model_name,
            encoder_name=config.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            num_classes=config.num_classes,
            activation=None  # 使用logits
        )
        model = model.to(device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

        # 记录模型信息到W&B
        wandb.config.update({"total_params": total_params}, allow_val_change=True)

        # 创建损失函数
        criterion = get_loss_function(config.loss_function)
        criterion = criterion.to(device)

        # 创建优化器
        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.get('weight_decay', 0)
            )
        elif config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.get('weight_decay', 1e-4)
            )
        elif config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # 创建学习率调度器
        scheduler = None
        if config.get('scheduler'):
            if config.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.num_epochs
                )
            elif config.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
            elif config.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.num_epochs // 3,
                    gamma=0.1
                )

        # 训练模型
        print("\nStarting training...")
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config.num_epochs,
            save_dir=config.output_dir,
            use_amp=config.get('use_amp', True),
            early_stopping_patience=config.get('early_stopping_patience', 10),
            log_wandb=True
        )

        # 记录最终结果
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best Validation Dice: {results['best_dice']:.4f}")
        print("="*70 + "\n")

        # W&B会自动记录这个值用于优化
        wandb.log({'validation_dice': results['best_dice']})


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with W&B Sweeps")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hpo_config.yaml",
        help="Path to HPO configuration file"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (overrides config file)"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Existing sweep ID to continue (optional)"
    )

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    # 准备W&B Sweep配置
    sweep_config = {
        'method': full_config.get('method', 'bayes'),
        'metric': full_config.get('metric', {'name': 'validation_dice', 'goal': 'maximize'}),
        'parameters': {}
    }

    # 添加早停配置（如果有）
    if 'early_terminate' in full_config:
        sweep_config['early_terminate'] = full_config['early_terminate']

    # 合并搜索参数和固定参数
    # 搜索参数
    for param_name, param_config in full_config.get('parameters', {}).items():
        sweep_config['parameters'][param_name] = param_config

    # 固定参数（作为constant）
    for param_name, param_value in full_config.get('fixed_params', {}).items():
        sweep_config['parameters'][param_name] = {'value': param_value}

    # 创建或获取Sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Continuing existing sweep: {sweep_id}")
    else:
        project_name = full_config.get('project', 'Osteoblast_HPO')
        entity = full_config.get('entity', None)

        print(f"\nInitializing W&B Sweep...")
        print(f"Project: {project_name}")
        print(f"Method: {sweep_config['method']}")
        print(f"Metric: {sweep_config['metric']}")

        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=project_name,
            entity=entity
        )
        print(f"\nSweep created! ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{entity if entity else 'your-username'}/{project_name}/sweeps/{sweep_id}")

    # 获取运行次数
    count = args.count if args.count is not None else full_config.get('run_config', {}).get('count', 10)

    print(f"\nStarting W&B Agent...")
    print(f"Will run {count} experiments")
    print("-"*70)

    # 启动W&B Agent
    wandb.agent(
        sweep_id=sweep_id,
        function=train_one_run,
        count=count
    )

    print("\n" + "="*70)
    print("HPO completed!")
    print(f"View results at: https://wandb.ai")
    print("="*70)


if __name__ == "__main__":
    main()
