#!/usr/bin/env python3
"""
W&B Sweep 结果分析脚本
Analyze and Visualize W&B Sweep Results

使用方法:
    python scripts/analyze_sweep.py --sweep_id YOUR_SWEEP_ID --project Osteoblast_HPO

作者: FTP Project
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path


def analyze_sweep(sweep_id: str, project: str, entity: str = None):
    """
    分析W&B Sweep结果

    Args:
        sweep_id: Sweep ID
        project: W&B项目名称
        entity: W&B实体名称（用户名或团队名）
    """
    # 初始化W&B API
    api = wandb.Api()

    # 获取sweep
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    print(f"Loading sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    # 获取所有runs
    runs = sweep.runs
    print(f"Found {len(runs)} runs")

    # 提取数据
    data = []
    for run in runs:
        # 基本信息
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
        }

        # 配置参数
        config = run.config
        for key, value in config.items():
            if key not in ['_wandb', 'wandb_version']:
                row[f'config_{key}'] = value

        # 指标
        summary = run.summary
        for key, value in summary.items():
            if not key.startswith('_'):
                row[f'metric_{key}'] = value

        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存到CSV
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"sweep_{sweep_id}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # 分析和可视化
    print("\n" + "="*70)
    print("SWEEP ANALYSIS")
    print("="*70)

    # 1. 最佳运行
    if 'metric_best_dice' in df.columns:
        best_run = df.loc[df['metric_best_dice'].idxmax()]
        print("\n🏆 Best Run:")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Run Name: {best_run['run_name']}")
        print(f"  Best Dice: {best_run['metric_best_dice']:.4f}")

        print("\n  Configuration:")
        config_cols = [col for col in df.columns if col.startswith('config_')]
        for col in config_cols:
            param_name = col.replace('config_', '')
            print(f"    {param_name}: {best_run[col]}")

    # 2. 统计摘要
    print("\n" + "-"*70)
    print("Statistical Summary:")
    print("-"*70)

    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    if metric_cols:
        print(df[metric_cols].describe())

    # 3. 生成可视化
    print("\nGenerating visualizations...")
    visualize_sweep(df, sweep_id, output_dir)

    print("\n" + "="*70)
    print("Analysis completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


def visualize_sweep(df: pd.DataFrame, sweep_id: str, output_dir: Path):
    """
    生成可视化图表

    Args:
        df: 包含sweep结果的DataFrame
        sweep_id: Sweep ID
        output_dir: 输出目录
    """
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. 参数重要性 - 箱线图
    config_cols = [col for col in df.columns if col.startswith('config_')]

    if 'metric_best_dice' in df.columns and config_cols:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hyperparameter Analysis - Sweep {sweep_id}', fontsize=16)

        # 选择重要的参数进行可视化
        important_params = ['config_model_name', 'config_learning_rate',
                           'config_batch_size', 'config_loss_function']

        for idx, param in enumerate(important_params):
            if param in df.columns:
                ax = axes[idx // 2, idx % 2]

                if df[param].dtype == 'object' or df[param].nunique() < 10:
                    # 类别型参数 - 箱线图
                    df.boxplot(column='metric_best_dice', by=param, ax=ax)
                    ax.set_xlabel(param.replace('config_', ''))
                    ax.set_ylabel('Best Dice Score')
                    ax.set_title('')
                else:
                    # 数值型参数 - 散点图
                    ax.scatter(df[param], df['metric_best_dice'], alpha=0.6)
                    ax.set_xlabel(param.replace('config_', ''))
                    ax.set_ylabel('Best Dice Score')
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'sweep_{sweep_id}_param_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / f'sweep_{sweep_id}_param_analysis.png'}")
        plt.close()

    # 2. 学习率 vs 性能
    if 'config_learning_rate' in df.columns and 'metric_best_dice' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['config_learning_rate'], df['metric_best_dice'],
                   c=df['metric_best_dice'], cmap='viridis', s=100, alpha=0.6)
        plt.colorbar(label='Best Dice Score')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Best Dice Score')
        plt.title(f'Learning Rate vs Performance - Sweep {sweep_id}')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'sweep_{sweep_id}_lr_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / f'sweep_{sweep_id}_lr_analysis.png'}")
        plt.close()

    # 3. 性能分布直方图
    if 'metric_best_dice' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['metric_best_dice'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Best Dice Score')
        plt.ylabel('Frequency')
        plt.title(f'Performance Distribution - Sweep {sweep_id}')
        plt.axvline(df['metric_best_dice'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["metric_best_dice"].mean():.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'sweep_{sweep_id}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / f'sweep_{sweep_id}_distribution.png'}")
        plt.close()

    # 4. 相关性热图
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if 'metric_' in col or 'config_' in col]

    if len(numeric_cols) > 2:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title(f'Parameter Correlation Matrix - Sweep {sweep_id}')
        plt.tight_layout()
        plt.savefig(output_dir / f'sweep_{sweep_id}_correlation.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / f'sweep_{sweep_id}_correlation.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B Sweep Results")
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B Sweep ID")
    parser.add_argument("--project", type=str, default="Osteoblast_HPO", help="W&B Project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B Entity (username or team)")

    args = parser.parse_args()

    analyze_sweep(
        sweep_id=args.sweep_id,
        project=args.project,
        entity=args.entity
    )


if __name__ == "__main__":
    main()
