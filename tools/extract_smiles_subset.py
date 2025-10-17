#!/usr/bin/env python3
"""
从 DrugLikeSMILSE-12B-427M 数据集中提取少量数据用于 debug
"""
import os
from datasets import load_from_disk, DatasetDict

def extract_subset(
    source_path: str = "data/DrugLikeSMILSE-12B-427M",
    target_path: str = "data/DrugLikeSMILSE-debug",
    train_size: int = 10000,
    val_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42
):
    """
    从源数据集提取子集
    
    Args:
        source_path: 源数据集路径
        target_path: 目标数据集路径
        train_size: 训练集样本数
        val_size: 验证集样本数
        test_size: 测试集样本数
        seed: 随机种子
    """
    print(f"从 {source_path} 加载数据集...")
    dataset = load_from_disk(source_path)
    
    print(f"原始数据集大小:")
    print(f"  train: {len(dataset['train']):,} 条")
    print(f"  validation: {len(dataset['validation']):,} 条")
    print(f"  test: {len(dataset['test']):,} 条")
    
    print(f"\n提取子集...")
    # 使用shuffle + select来随机采样
    subset = DatasetDict({
        'train': dataset['train'].shuffle(seed=seed).select(range(train_size)),
        'validation': dataset['validation'].shuffle(seed=seed).select(range(val_size)),
        'test': dataset['test'].shuffle(seed=seed).select(range(test_size))
    })
    
    print(f"\n子集大小:")
    print(f"  train: {len(subset['train']):,} 条")
    print(f"  validation: {len(subset['validation']):,} 条")
    print(f"  test: {len(subset['test']):,} 条")
    
    # 显示一些示例
    print(f"\n训练集示例:")
    for i in range(min(3, len(subset['train']))):
        example = subset['train'][i]
        smiles = example['input']
        n_tokens = example['n_tokens']
        print(f"  [{i}] tokens={n_tokens}, SMILES={smiles[:80]}{'...' if len(smiles) > 80 else ''}")
    
    print(f"\n保存到 {target_path}...")
    os.makedirs(os.path.dirname(target_path) if os.path.dirname(target_path) else '.', exist_ok=True)
    subset.save_to_disk(target_path)
    
    print(f"\n完成！子集已保存到: {target_path}")
    print(f"使用方法: raw_data_path={target_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从SMILES数据集提取debug子集")
    parser.add_argument(
        "--source",
        default="data/DrugLikeSMILSE-12B-427M",
        help="源数据集路径"
    )
    parser.add_argument(
        "--target",
        default="data/DrugLikeSMILSE-debug",
        help="目标数据集路径"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10000,
        help="训练集样本数"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="验证集样本数"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="测试集样本数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    extract_subset(
        source_path=args.source,
        target_path=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed
    )