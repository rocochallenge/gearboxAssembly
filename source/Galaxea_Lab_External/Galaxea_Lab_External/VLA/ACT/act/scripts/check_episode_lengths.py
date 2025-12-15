import h5py
from pathlib import Path

def check_episode_lengths(dataset_dir):
    """检查数据集中所有 episode 的长度"""
    dataset_path = Path(dataset_dir)
    episode_files = sorted(dataset_path.glob('episode_*.hdf5'))
    
    if not episode_files:
        print(f"❌ 在 {dataset_dir} 中没有找到 episode 文件")
        return
    
    print(f"📊 检查 {len(episode_files)} 个 episode 文件的长度...\n")
    
    lengths = []
    for episode_file in episode_files:
        with h5py.File(episode_file, 'r') as f:
            # 检查 action 或 qpos 的长度
            if 'action' in f:
                length = f['action'].shape[0]
            elif 'observations/qpos' in f:
                length = f['observations/qpos'].shape[0]
            else:
                print(f"⚠️  {episode_file.name}: 找不到 action 或 qpos 数据")
                continue
            
            lengths.append(length)
            print(f"  {episode_file.name}: {length} 步")
    
    if lengths:
        print(f"\n📈 统计信息:")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
        print(f"  平均长度: {sum(lengths)/len(lengths):.1f}")
        print(f"\n💡 建议在 constants.py 中设置:")
        print(f"  'episode_len': {max(lengths)},")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = '/media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos_filtered'
    
    check_episode_lengths(dataset_dir)