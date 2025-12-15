#!/usr/bin/env python3
"""
为 h5 文件添加 sim 属性
"""

import h5py
from pathlib import Path
import argparse


def add_sim_attribute(h5_path, is_sim=False):
    """
    为单个 h5 文件添加 sim 属性
    
    Args:
        h5_path: h5 文件路径
        is_sim: True 表示仿真数据，False 表示真实机器人数据
    
    Returns:
        bool: 是否成功
    """
    try:
        with h5py.File(h5_path, 'r+') as f:
            if 'sim' in f.attrs:
                current_value = f.attrs['sim']
                print(f"  ℹ️  已有 sim 属性: {current_value}")
                if current_value != is_sim:
                    f.attrs['sim'] = is_sim
                    print(f"  ✅ 更新为: {is_sim}")
                    return True
                return False
            else:
                f.attrs['sim'] = is_sim
                print(f"  ✅ 添加 sim 属性: {is_sim}")
                return True
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        return False


def batch_add_sim_attribute(directory, is_sim=False):
    """
    批量为目录下所有 h5 文件添加 sim 属性
    
    Args:
        directory: 目录路径
        is_sim: True 表示仿真数据，False 表示真实机器人数据
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ 错误: 目录不存在 - {directory}")
        return
    
    # 查找所有 h5 文件
    h5_files = sorted(list(dir_path.glob('*.h5')) + list(dir_path.glob('*.hdf5')))
    
    if not h5_files:
        print(f"⚠️  在目录中未找到 h5 文件: {directory}")
        return
    
    print(f"🔍 找到 {len(h5_files)} 个 h5 文件")
    print(f"📂 目录: {directory}")
    print(f"🏷️  设置 sim = {is_sim} ({'仿真数据' if is_sim else '真实机器人数据'})\n")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for idx, h5_file in enumerate(h5_files, 1):
        print(f"[{idx}/{len(h5_files)}] {h5_file.name}")
        
        result = add_sim_attribute(h5_file, is_sim)
        if result is None:
            failed_count += 1
        elif result:
            success_count += 1
        else:
            skipped_count += 1
    
    print(f"\n{'='*80}")
    print(f"📊 处理完成!")
    print(f"{'='*80}")
    print(f"✅ 成功添加/更新: {success_count} 个文件")
    print(f"⏭️  跳过（已有正确值）: {skipped_count} 个文件")
    print(f"❌ 失败: {failed_count} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description='为 h5 文件添加 sim 属性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 为真实机器人数据添加属性（sim=False）:
   python add_sim_attr.py --dir /path/to/dataset

2. 为仿真数据添加属性（sim=True）:
   python add_sim_attr.py --dir /path/to/dataset --sim

3. 处理单个文件:
   python add_sim_attr.py --file episode_0.hdf5

示例:
   python add_sim_attr.py --dir /media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos
        """
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='目录路径（批量处理）'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='单个文件路径'
    )
    
    parser.add_argument(
        '--sim',
        action='store_true',
        help='设置为仿真数据（sim=True），默认为真实机器人数据（sim=False）'
    )
    
    args = parser.parse_args()
    
    if args.dir:
        batch_add_sim_attribute(args.dir, is_sim=args.sim)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在 - {args.file}")
            return
        print(f"处理文件: {file_path.name}")
        add_sim_attribute(file_path, is_sim=args.sim)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
