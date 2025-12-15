#!/usr/bin/env python3
"""
检查指定目录下所有 h5 文件中的 action 数据是否全为 0
# 检查当前目录（递归）
python check_zero_actions.py .

# 检查指定目录
python check_zero_actions.py /path/to/datasets

# 只检查当前目录，不递归子目录
python check_zero_actions.py /path/to/datasets --no-recursive
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def check_action_zeros(h5_path):
    """
    检查单个 h5 文件中的 action 是否全为 0
    
    Args:
        h5_path: h5 文件路径
    
    Returns:
        dict: 包含检查结果的字典
    """
    result = {
        'path': str(h5_path),
        'has_action': False,
        'all_zeros': False,
        'shape': None,
        'non_zero_count': 0,
        'total_elements': 0,
        'min_val': None,
        'max_val': None,
        'mean_val': None,
        'error': None
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 检查是否存在 action 数据集或 actions 组
            if 'action' in f and isinstance(f['action'], h5py.Dataset):
                # 情况1: 直接有 action 数据集（已合并的）
                result['has_action'] = True
                action_data = f['action'][:]
                result['shape'] = action_data.shape
                result['total_elements'] = action_data.size
                
                # 统计非零值数量
                result['non_zero_count'] = np.count_nonzero(action_data)
                result['all_zeros'] = (result['non_zero_count'] == 0)
                
                # 统计数值范围
                result['min_val'] = float(np.min(action_data))
                result['max_val'] = float(np.max(action_data))
                result['mean_val'] = float(np.mean(action_data))
                
            elif 'actions' in f and isinstance(f['actions'], h5py.Group):
                # 情况2: 有 actions 组，包含四个数据集
                result['has_action'] = True
                
                # 读取四个 action 数据集
                action_keys = ['left_arm_action', 'left_gripper_action', 
                              'right_arm_action', 'right_gripper_action']
                
                missing_keys = [key for key in action_keys if key not in f['actions']]
                if missing_keys:
                    result['error'] = f"actions 组中缺少数据集: {missing_keys}"
                    return result
                
                # 合并所有 action 数据
                all_actions = []
                for key in action_keys:
                    data = f[f'actions/{key}'][:]
                    # 确保是二维数组
                    if data.ndim == 1:
                        data = data[:, np.newaxis]
                    all_actions.append(data)
                
                action_data = np.concatenate(all_actions, axis=1)
                result['shape'] = action_data.shape
                result['total_elements'] = action_data.size
                
                # 统计非零值数量
                result['non_zero_count'] = np.count_nonzero(action_data)
                result['all_zeros'] = (result['non_zero_count'] == 0)
                
                # 统计数值范围
                result['min_val'] = float(np.min(action_data))
                result['max_val'] = float(np.max(action_data))
                result['mean_val'] = float(np.mean(action_data))
                
            else:
                result['error'] = "文件中不存在 'action' 数据集或 'actions' 组"
                return result
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def scan_directory(directory, recursive=True):
    """
    扫描目录下所有 h5 文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归扫描子目录
    
    Returns:
        list: 检查结果列表
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ 错误: 目录不存在 - {directory}")
        return []
    
    # 查找所有 h5 文件
    if recursive:
        h5_files = list(dir_path.rglob('*.h5')) + list(dir_path.rglob('*.hdf5'))
    else:
        h5_files = list(dir_path.glob('*.h5')) + list(dir_path.glob('*.hdf5'))
    
    if not h5_files:
        print(f"⚠️  在目录中未找到 h5 文件: {directory}")
        return []
    
    print(f"🔍 找到 {len(h5_files)} 个 h5 文件，开始检查...\n")
    
    results = []
    for h5_file in sorted(h5_files):
        print(f"检查: {h5_file.name}")
        result = check_action_zeros(h5_file)
        results.append(result)
        
        # 打印简要信息
        if result['error']:
            print(f"  ⚠️  {result['error']}")
        elif result['has_action']:
            if result['all_zeros']:
                print(f"  ❌ 全为 0! shape: {result['shape']}")
            else:
                print(f"  ✅ 正常 - 非零值: {result['non_zero_count']}/{result['total_elements']}, "
                      f"范围: [{result['min_val']:.6f}, {result['max_val']:.6f}]")
        print()
    
    return results


def print_summary(results):
    """打印统计摘要"""
    print("\n" + "="*80)
    print("📊 检查结果汇总")
    print("="*80)
    
    total = len(results)
    has_action = sum(1 for r in results if r['has_action'])
    all_zeros = sum(1 for r in results if r['all_zeros'])
    has_error = sum(1 for r in results if r['error'])
    
    print(f"\n总文件数: {total}")
    print(f"包含 action 数据: {has_action}")
    print(f"全为 0 的文件: {all_zeros}")
    print(f"检查出错的文件: {has_error}")
    
    if all_zeros > 0:
        print(f"\n❌ 以下 {all_zeros} 个文件的 action 数据全为 0:")
        for r in results:
            if r['all_zeros']:
                print(f"  - {r['path']} (shape: {r['shape']})")
    
    if has_error > 0:
        print(f"\n⚠️  以下 {has_error} 个文件检查时出错:")
        for r in results:
            if r['error']:
                print(f"  - {r['path']}")
                print(f"    错误: {r['error']}")
    
    normal_count = has_action - all_zeros
    if normal_count > 0:
        print(f"\n✅ {normal_count} 个文件的 action 数据正常")


def main():
    parser = argparse.ArgumentParser(
        description='检查目录下所有 h5 文件中的 action 数据是否全为 0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查当前目录
  python check_zero_actions.py .
  
  # 检查指定目录（递归）
  python check_zero_actions.py /path/to/datasets
  
  # 只检查当前目录（不递归）
  python check_zero_actions.py /path/to/datasets --no-recursive
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='要检查的目录路径'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归扫描子目录'
    )
    
    args = parser.parse_args()
    
    print(f"🔍 开始扫描目录: {args.directory}")
    print(f"递归模式: {'否' if args.no_recursive else '是'}\n")
    
    results = scan_directory(args.directory, recursive=not args.no_recursive)
    
    if results:
        print_summary(results)
        
        # 返回退出码：如果有全为0的文件，返回1
        all_zeros_count = sum(1 for r in results if r['all_zeros'])
        if all_zeros_count > 0:
            exit(1)
    else:
        print("未找到可检查的文件")
        exit(1)


if __name__ == "__main__":
    main()
