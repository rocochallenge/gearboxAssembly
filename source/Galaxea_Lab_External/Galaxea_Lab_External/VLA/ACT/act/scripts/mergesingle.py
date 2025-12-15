#!/usr/bin/env python3
"""
将 h5 文件中的四个 actions 数据拼接成一个 14 维数据:
- left_arm_action (6维) + left_gripper_action (1维) + right_arm_action (6维) + right_gripper_action (1维)
合并后的数据命名为 action 并放在根目录下
删除原有的 actions 组
同时将分离的关节数据合并成 qpos (14维)
修改 INPUT_H5_PATH 和 OUTPUT_H5_PATH 来指定输入输出文件路径
"""

import h5py
import numpy as np
import shutil
from pathlib import Path

# ==================== 在这里修改文件路径 ====================
INPUT_H5_PATH = "/media/mldadmin/home/s123mdg31_14/datasets/gearbox_assembly_demos/data_20251127_212217.hdf5"
OUTPUT_H5_PATH = "/media/mldadmin/home/s123mdg31_14/act/datasets/gearbox_assembly_demos/episode_0.hdf5"
# ===========================================================


def merge_actions(input_path, output_path):
    """
    将 h5 文件中的四个 action 数据拼接成一个 14 维数据:
    - left_arm_action (6维) + left_gripper_action (1维) + right_arm_action (6维) + right_gripper_action (1维)
    并将合并后的数据命名为 action 放在根目录下
    删除原有的 actions 组
    同时在 observations 下创建 images 组,并将 rgb 图像数据移入其中
    同时将分离的关节数据合并成 qpos (14维)
    """
    print(f"读取文件: {input_path}")
    
    # 检查输入输出是否为同一文件
    input_path_resolved = Path(input_path).resolve()
    output_path_resolved = Path(output_path).resolve()
    is_same_file = input_path_resolved == output_path_resolved
    
    if is_same_file:
        print(f"ℹ️  输入输出为同一文件,将直接修改原文件")
    else:
        print(f"ℹ️  输入输出为不同文件,将复制后再修改")
    
    # 先检查输入文件的结构
    print("\n🔍 检查文件结构...")
    process_actions = False
    process_images = False
    process_qpos = False
    
    with h5py.File(input_path, 'r') as f:
        # 检查是否需要处理 actions
        if 'actions' not in f:
            print(f"ℹ️  文件中不存在 'actions' 组,跳过 actions 处理")
        else:
            # 检查四个action数据集是否都存在
            required_actions = [
                'actions/left_arm_action',
                'actions/left_gripper_action',
                'actions/right_arm_action',
                'actions/right_gripper_action'
            ]
            missing_actions = [key for key in required_actions if key not in f]
            
            if missing_actions:
                print(f"ℹ️  缺少以下 action 数据集,跳过 actions 处理:")
                for key in missing_actions:
                    print(f"     - {key}")
            else:
                # 检查是否已经存在根目录的 action 数据集
                if 'action' in f:
                    print(f"⚠️  文件根目录已存在 'action' 数据集")
                    print(f"   action shape: {f['action'].shape}")
                    user_input = input("   是否要重新生成 action? (y/n): ").strip().lower()
                    if user_input == 'y':
                        process_actions = True
                else:
                    process_actions = True
        
        # 检查是否需要处理 images
        if 'observations' in f:
            rgb_keys = ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
            has_rgb = any(f'observations/{key}' in f for key in rgb_keys)
            if has_rgb:
                if 'observations/images' in f:
                    # 检查是否所有图像都已在 images 组中
                    all_in_images = all(f'observations/images/{key}' in f for key in rgb_keys if f'observations/{key}' not in f)
                    if not all_in_images:
                        process_images = True
                    else:
                        print(f"ℹ️  所有图像数据已在 observations/images 中,跳过图像重组")
                else:
                    process_images = True
            else:
                print(f"ℹ️  文件中不存在 rgb 图像数据,跳过图像处理")
        else:
            print(f"ℹ️  文件中不存在 observations 组,跳过图像处理")
        
        # 检查是否需要处理 qpos
        if 'observations' in f:
            qpos_keys = [
                'observations/left_arm_joint_pos',
                'observations/left_gripper_joint_pos',
                'observations/right_arm_joint_pos',
                'observations/right_gripper_joint_pos'
            ]
            has_all_qpos_parts = all(key in f for key in qpos_keys)
            
            if has_all_qpos_parts:
                if 'observations/qpos' in f:
                    print(f"⚠️  observations/qpos 已存在: {f['observations/qpos'].shape}")
                    user_input = input("   是否要重新生成 qpos? (y/n): ").strip().lower()
                    if user_input == 'y':
                        process_qpos = True
                else:
                    process_qpos = True
            else:
                missing = [key for key in qpos_keys if key not in f]
                if missing:
                    print(f"ℹ️  缺少关节数据,跳过 qpos 处理:")
                    for key in missing:
                        print(f"     - {key}")
        
        # 如果没有任何需要处理的内容
        if not process_actions and not process_images and not process_qpos:
            print(f"\n⚠️  没有需要处理的内容")
            return False
        
        print(f"\n✅ 文件结构检查通过")
        if process_actions:
            print(f"   ✓ 将处理 actions 数据合并")
        if process_images:
            print(f"   ✓ 将处理图像数据重组")
        if process_qpos:
            print(f"   ✓ 将处理关节数据合并为 qpos")
    
    # 如果不是同一文件,则复制
    if not is_same_file:
        print(f"\n📋 复制文件到: {output_path}")
        shutil.copy2(input_path, output_path)
        file_to_modify = output_path
    else:
        print(f"\n✏️  直接修改文件: {input_path}")
        file_to_modify = input_path
    
    # 打开文件进行修改
    with h5py.File(file_to_modify, 'r+') as f:
        # 处理 actions 数据
        if process_actions:
            print("\n📊 处理 actions 数据...")
            print("原始 actions 结构:")
            print(f"  left_arm_action shape: {f['actions/left_arm_action'].shape}")
            print(f"  left_gripper_action shape: {f['actions/left_gripper_action'].shape}")
            print(f"  right_arm_action shape: {f['actions/right_arm_action'].shape}")
            print(f"  right_gripper_action shape: {f['actions/right_gripper_action'].shape}")
            
            # 读取四个 action 数据
            left_arm_action = f['actions/left_arm_action'][:]  # (N, 6)
            left_gripper_action = f['actions/left_gripper_action'][:]  # (N, 1)
            right_arm_action = f['actions/right_arm_action'][:]  # (N, 6)
            right_gripper_action = f['actions/right_gripper_action'][:]  # (N, 1)
            
            print(f"\n读取数据:")
            print(f"  left_arm_action: {left_arm_action.shape}, dtype: {left_arm_action.dtype}")
            print(f"  left_gripper_action: {left_gripper_action.shape}, dtype: {left_gripper_action.dtype}")
            print(f"  right_arm_action: {right_arm_action.shape}, dtype: {right_arm_action.dtype}")
            print(f"  right_gripper_action: {right_gripper_action.shape}, dtype: {right_gripper_action.dtype}")
            
            # 确保 gripper action 是 (N, 1) 形状
            if left_gripper_action.ndim == 1:
                left_gripper_action = left_gripper_action[:, np.newaxis]
            if right_gripper_action.ndim == 1:
                right_gripper_action = right_gripper_action[:, np.newaxis]
            
            print(f"\n维度校验后:")
            print(f"  left_gripper_action: {left_gripper_action.shape}")
            print(f"  right_gripper_action: {right_gripper_action.shape}")
            
            # 拼接成 14 维数据 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            merged_action = np.concatenate([
                left_arm_action,      # (N, 6)
                left_gripper_action,  # (N, 1)
                right_arm_action,     # (N, 6)
                right_gripper_action  # (N, 1)
            ], axis=1)
            
            print(f"\n拼接后:")
            print(f"  merged_action: {merged_action.shape}, dtype: {merged_action.dtype}")
            print(f"  第一帧示例: {merged_action[0]}")
            print(f"  结构: left_arm(0-5) + left_gripper(6) + right_arm(7-12) + right_gripper(13)")
            
            # 在根目录下创建 action 数据集
            if 'action' in f:
                del f['action']
            
            f.create_dataset('action', data=merged_action, dtype='float32')
            
            print(f"\n✅ 成功在根目录创建 'action' 数据集 (shape: {merged_action.shape})")
            
            # 删除原有的 actions 组
            if 'actions' in f:
                del f['actions']
                print(f"✅ 已删除原有的 'actions' 组")
        
        # 处理 observations 中的图像数据
        if process_images:
            print(f"\n📸 处理 observations 中的图像数据...")
            if 'observations' in f:
                # 创建 images 组
                if 'observations/images' not in f:
                    f['observations'].create_group('images')
                    print(f"✅ 创建 observations/images 组")
                
                # 移动 rgb 图像数据
                rgb_keys = ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
                for key in rgb_keys:
                    obs_key = f'observations/{key}'
                    if obs_key in f:
                        # 读取数据
                        data = f[obs_key][:]
                        print(f"  移动 {key}: {data.shape}")
                        
                        # 在 images 组中创建数据集
                        img_key = f'observations/images/{key}'
                        if img_key in f:
                            del f[img_key]
                        f['observations/images'].create_dataset(key, data=data, dtype=data.dtype)
                        
                        # 删除原位置的数据集
                        del f[obs_key]
                        print(f"  ✅ {key} 已移动到 observations/images/{key}")
        
        # 处理 qpos 数据
        if process_qpos:
            print(f"\n🤖 处理关节数据合并为 qpos 和 qvel...")
            
            # 读取位置数据
            left_arm = f['observations/left_arm_joint_pos'][:]  # (N, 6)
            left_gripper = f['observations/left_gripper_joint_pos'][:]  # (N, 1)
            right_arm = f['observations/right_arm_joint_pos'][:]  # (N, 6)
            right_gripper = f['observations/right_gripper_joint_pos'][:]  # (N, 1)
            
            print(f"原始关节位置数据:")
            print(f"  left_arm_joint_pos: {left_arm.shape}, dtype: {left_arm.dtype}")
            print(f"  left_gripper_joint_pos: {left_gripper.shape}, dtype: {left_gripper.dtype}")
            print(f"  right_arm_joint_pos: {right_arm.shape}, dtype: {right_arm.dtype}")
            print(f"  right_gripper_joint_pos: {right_gripper.shape}, dtype: {right_gripper.dtype}")
            
            # 确保 gripper 位置是 (N, 1) 形状
            if left_gripper.ndim == 1:
                left_gripper = left_gripper[:, np.newaxis]
            if right_gripper.ndim == 1:
                right_gripper = right_gripper[:, np.newaxis]
            
            # 读取速度数据
            left_arm_vel = f['observations/left_arm_joint_vel'][:]  # (N, 6)
            left_gripper_vel = f['observations/left_gripper_joint_vel'][:]  # (N, 1)
            right_arm_vel = f['observations/right_arm_joint_vel'][:]  # (N, 6)
            right_gripper_vel = f['observations/right_gripper_joint_vel'][:]  # (N, 1)
            
            print(f"\n原始关节速度数据:")
            print(f"  left_arm_joint_vel: {left_arm_vel.shape}, dtype: {left_arm_vel.dtype}")
            print(f"  left_gripper_joint_vel: {left_gripper_vel.shape}, dtype: {left_gripper_vel.dtype}")
            print(f"  right_arm_joint_vel: {right_arm_vel.shape}, dtype: {right_arm_vel.dtype}")
            print(f"  right_gripper_joint_vel: {right_gripper_vel.shape}, dtype: {right_gripper_vel.dtype}")
            
            # 确保 gripper 速度是 (N, 1) 形状
            if left_gripper_vel.ndim == 1:
                left_gripper_vel = left_gripper_vel[:, np.newaxis]
            if right_gripper_vel.ndim == 1:
                right_gripper_vel = right_gripper_vel[:, np.newaxis]
            
            # 合并成 qpos (N, 14) 并转换为 float32
            # qpos = [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            qpos = np.concatenate([
                left_arm,      # (N, 6)
                left_gripper,  # (N, 1)
                right_arm,     # (N, 6)
                right_gripper  # (N, 1)
            ], axis=1).astype(np.float32)  # 结果: (N, 14), dtype: float32
            
            # 合并成 qvel (N, 14) 并转换为 float32
            # qvel = [left_arm_vel(6) + left_gripper_vel(1) + right_arm_vel(6) + right_gripper_vel(1)]
            qvel = np.concatenate([
                left_arm_vel,      # (N, 6)
                left_gripper_vel,  # (N, 1)
                right_arm_vel,     # (N, 6)
                right_gripper_vel  # (N, 1)
            ], axis=1).astype(np.float32)  # 结果: (N, 14), dtype: float32
            
            print(f"\n拼接后:")
            print(f"  qpos: {qpos.shape}, dtype: {qpos.dtype}")
            print(f"  qvel: {qvel.shape}, dtype: {qvel.dtype}")
            print(f"\n  qpos 第一帧:")
            print(f"    left_arm (0-5): {qpos[0, 0:6]}")
            print(f"    left_gripper (6): {qpos[0, 6]}")
            print(f"    right_arm (7-12): {qpos[0, 7:13]}")
            print(f"    right_gripper (13): {qpos[0, 13]}")
            print(f"\n  qvel 第一帧:")
            print(f"    left_arm_vel (0-5): {qvel[0, 0:6]}")
            print(f"    left_gripper_vel (6): {qvel[0, 6]}")
            print(f"    right_arm_vel (7-12): {qvel[0, 7:13]}")
            print(f"    right_gripper_vel (13): {qvel[0, 13]}")
            
            # 在 observations 下创建 qpos 数据集
            if 'observations/qpos' in f:
                del f['observations/qpos']
            
            f['observations'].create_dataset('qpos', data=qpos, dtype='float32')
            
            print(f"\n✅ 成功创建 'observations/qpos' 数据集 (shape: {qpos.shape}, dtype: float32)")
            
            # 创建 qvel 数据集
            if 'observations/qvel' in f:
                del f['observations/qvel']
            
            f['observations'].create_dataset('qvel', data=qvel, dtype='float32')
            print(f"✅ 成功创建 'observations/qvel' 数据集 (shape: {qvel.shape}, dtype: float32)")
            print(f"   - 包含左右臂关节速度 + 左右夹爪速度")
            
            # 删除原始的关节位置和速度数据
            keys_to_delete = [
                'left_arm_joint_pos', 'left_gripper_joint_pos',
                'right_arm_joint_pos', 'right_gripper_joint_pos',
                'left_arm_joint_vel', 'left_gripper_joint_vel',
                'right_arm_joint_vel', 'right_gripper_joint_vel'
            ]
            for key in keys_to_delete:
                if key in f['observations']:
                    del f['observations'][key]
                    print(f"  🗑️  删除 observations/{key}")
        
        # 清理 observations 中除了 images、qpos、qvel 之外的所有数据
        if 'observations' in f:
            print(f"\n🧹 清理 observations 组...")
            keep_keys = {'images', 'qpos', 'qvel'}
            keys_to_remove = [key for key in f['observations'].keys() if key not in keep_keys]
            
            for key in keys_to_remove:
                del f['observations'][key]
                print(f"  🗑️  删除 observations/{key}")
            
            if keys_to_remove:
                print(f"✅ 已清理 {len(keys_to_remove)} 个多余数据")
            else:
                print(f"ℹ️  无需清理，observations 中只包含必要数据")
        
        print(f"\n最终文件结构:")
        print(f"根目录:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  - {key}: {f[key].shape}")
            else:
                print(f"  - {key}/ (Group)")
        
        if 'observations' in f:
            print(f"\nobservations 组:")
            for key in f['observations'].keys():
                if isinstance(f['observations'][key], h5py.Dataset):
                    print(f"  - {key}: {f['observations'][key].shape}")
                else:
                    print(f"  - {key}/ (Group)")
            
            if 'observations/images' in f:
                print(f"\nobservations/images 组:")
                for key in f['observations/images'].keys():
                    if isinstance(f['observations/images'][key], h5py.Dataset):
                        print(f"  - {key}: {f['observations/images'][key].shape}")
    
    if is_same_file:
        print(f"\n✅ 处理完成! 已直接修改原文件: {input_path}")
    else:
        print(f"\n✅ 处理完成! 输出文件: {output_path}")
        print(f"   原始文件保持不变: {input_path}")
    
    print(f"\n说明:")
    if process_actions:
        print(f"  - 在根目录添加了 'action' (14维) = [left_arm_action(6) + left_gripper_action(1) + right_arm_action(6) + right_gripper_action(1)]")
        print(f"  - action 结构: 索引0-5(左臂) + 索引6(左夹爪) + 索引7-12(右臂) + 索引13(右夹爪)")
        print(f"  - 已删除原有的 'actions' 组及其所有数据")
    if process_images:
        print(f"  - 在 observations 下创建了 images 组")
        print(f"  - head_rgb, left_hand_rgb, right_hand_rgb 已移动到 observations/images/")
    if process_qpos:
        print(f"  - 在 observations 下添加了 'qpos' (14维) = [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]")
        print(f"  - qpos 结构: 索引0-5(左臂6关节) + 索引6(左夹爪) + 索引7-12(右臂6关节) + 索引13(右夹爪)")
        print(f"  - 同时创建了 'qvel' (14维) = [left_arm_vel(6) + left_gripper_vel(1) + right_arm_vel(6) + right_gripper_vel(1)]")
    
    return True


def main():
    try:
        # 检查输入文件是否存在
        if not Path(INPUT_H5_PATH).exists():
            print(f"❌ 错误: 输入文件不存在 - {INPUT_H5_PATH}")
            return
        
        # 执行合并
        if not merge_actions(INPUT_H5_PATH, OUTPUT_H5_PATH):
            print(f"⚠️  未执行任何操作")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
