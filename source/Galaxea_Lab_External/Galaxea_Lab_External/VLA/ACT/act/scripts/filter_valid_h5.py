#!/usr/bin/env python3
"""
筛选并复制包含有效 action 数据（非全零）的 h5 文件
将筛选后的文件重新命名为 episode_0.hdf5 到 episode_n.hdf5
# 1. 先测试（只检查，不复制）
python filter_valid_h5.py \
    --input-dir /media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos \
    --output-dir /media/mldadmin/home/s123mdg31_14/datasets/gearbox_assembly_demos_filtered \
    --dry-run

# 2. 实际执行筛选和复制
python filter_valid_h5.py \
    --input-dir /media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos \
    --output-dir /media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos_filtered
"""

import h5py
import numpy as np
from pathlib import Path
import shutil
import argparse


def check_action_valid(h5_path):
    """
    检查 h5 文件的 action 数据是否有效（非全零）
    
    Args:
        h5_path: h5 文件路径
    
    Returns:
        tuple: (is_valid, error_message)
               is_valid: True 表示有效（非全零），False 表示全零或出错
               error_message: 如果出错，返回错误信息；否则为 None
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # 情况1: 直接有 action 数据集（已合并的）
            if 'action' in f and isinstance(f['action'], h5py.Dataset):
                action_data = f['action'][:]
                non_zero_count = np.count_nonzero(action_data)
                return (non_zero_count > 0, None)
            
            # 情况2: 有 actions 组，包含四个数据集
            elif 'actions' in f and isinstance(f['actions'], h5py.Group):
                action_keys = ['left_arm_action', 'left_gripper_action', 
                              'right_arm_action', 'right_gripper_action']
                
                missing_keys = [key for key in action_keys if key not in f['actions']]
                if missing_keys:
                    return (False, f"actions 组中缺少数据集: {missing_keys}")
                
                # 检查所有 action 数据
                all_actions = []
                for key in action_keys:
                    data = f[f'actions/{key}'][:]
                    if data.ndim == 1:
                        data = data[:, np.newaxis]
                    all_actions.append(data)
                
                action_data = np.concatenate(all_actions, axis=1)
                non_zero_count = np.count_nonzero(action_data)
                return (non_zero_count > 0, None)
            
            else:
                return (False, "文件中不存在 'action' 数据集或 'actions' 组")
                
    except Exception as e:
        return (False, f"读取文件出错: {str(e)}")


def filter_and_copy_valid_files(input_dir, output_dir, dry_run=False):
    """
    筛选有效的 h5 文件并复制到输出目录
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        dry_run: 如果为 True，只检查不复制
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 错误: 输入目录不存在 - {input_dir}")
        return
    
    # 查找所有 h5 文件
    h5_files = sorted(list(input_path.glob('*.h5')) + list(input_path.glob('*.hdf5')))
    
    if not h5_files:
        print(f"⚠️  在目录中未找到 h5 文件: {input_dir}")
        return
    
    print(f"🔍 找到 {len(h5_files)} 个 h5 文件")
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    if dry_run:
        print(f"🧪 [测试模式] 只检查，不复制文件")
    print()
    
    valid_files = []
    invalid_files = []
    error_files = []
    
    # 第一步：检查所有文件
    print("=" * 80)
    print("第一步：检查所有文件的 action 数据")
    print("=" * 80)
    
    for idx, h5_file in enumerate(h5_files, 1):
        print(f"[{idx}/{len(h5_files)}] 检查: {h5_file.name} ... ", end='', flush=True)
        
        is_valid, error_msg = check_action_valid(h5_file)
        
        if error_msg:
            print(f"❌ 错误")
            print(f"     原因: {error_msg}")
            error_files.append((h5_file, error_msg))
        elif is_valid:
            print(f"✅ 有效")
            valid_files.append(h5_file)
        else:
            print(f"⚠️  全零")
            invalid_files.append(h5_file)
    
    # 显示统计信息
    print(f"\n" + "=" * 80)
    print("检查结果统计")
    print("=" * 80)
    print(f"✅ 有效文件: {len(valid_files)}")
    print(f"⚠️  全零文件: {len(invalid_files)}")
    print(f"❌ 错误文件: {len(error_files)}")
    print(f"📊 总计: {len(h5_files)}")
    
    if invalid_files:
        print(f"\n全零文件列表:")
        for f in invalid_files:
            print(f"  - {f.name}")
    
    if error_files:
        print(f"\n错误文件列表:")
        for f, err in error_files:
            print(f"  - {f.name}: {err}")
    
    if not valid_files:
        print(f"\n⚠️  没有找到有效的文件！")
        return
    
    # 第二步：复制有效文件
    if dry_run:
        print(f"\n🧪 [测试模式] 将复制 {len(valid_files)} 个有效文件")
        print("\n预览输出文件名:")
        for idx, h5_file in enumerate(valid_files):
            print(f"  {h5_file.name} -> episode_{idx}.hdf5")
        print(f"\n💡 移除 --dry-run 参数来实际执行复制")
        return
    
    print(f"\n" + "=" * 80)
    print(f"第二步：复制有效文件并重命名")
    print("=" * 80)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for idx, h5_file in enumerate(valid_files):
        output_file = output_path / f"episode_{idx}.hdf5"
        
        try:
            print(f"[{idx + 1}/{len(valid_files)}] 复制: {h5_file.name} -> {output_file.name} ... ", 
                  end='', flush=True)
            
            shutil.copy2(h5_file, output_file)
            
            # 验证复制后的文件
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ ({file_size:.1f} MB)")
                success_count += 1
            else:
                print(f"❌ 复制失败")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            failed_count += 1
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print("复制完成!")
    print("=" * 80)
    print(f"✅ 成功复制: {success_count} 个文件")
    print(f"❌ 复制失败: {failed_count} 个文件")
    print(f"📂 输出目录: {output_path.absolute()}")
    print(f"\n输出文件命名: episode_0.hdf5 到 episode_{len(valid_files)-1}.hdf5")


def main():
    parser = argparse.ArgumentParser(
        description='筛选并复制包含有效 action 数据（非全零）的 h5 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 先测试（不实际复制）:
   python filter_valid_h5.py --input-dir /path/to/input --output-dir /path/to/output --dry-run

2. 实际执行筛选和复制:
   python filter_valid_h5.py --input-dir /path/to/input --output-dir /path/to/output

3. 示例:
   python filter_valid_h5.py \\
       --input-dir /media/mldadmin/home/s123mdg31_14/datasets/gearbox_assembly_demos \\
       --output-dir /media/mldadmin/home/s123mdg31_14/datasets/gearbox_assembly_demos_filtered

说明:
- 会检查 action 或 actions 组中的数据是否全为 0
- 只复制非全零的文件到输出目录
- 输出文件自动重命名为 episode_0.hdf5, episode_1.hdf5, ..., episode_n.hdf5
- 使用 --dry-run 可以先预览结果，不实际复制文件
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='输入目录路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='测试模式：只检查不复制文件'
    )
    
    args = parser.parse_args()
    
    try:
        filter_and_copy_valid_files(args.input_dir, args.output_dir, args.dry_run)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
