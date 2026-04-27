from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 定义参数
    mobiman_pkg_dir = get_package_share_directory('mobiman')
    urdf_file_path = os.path.join(mobiman_pkg_dir, 'urdf', 'R1_Lite', 'urdf', 'mmp_revB_invconfig_upright_a1x.urdf')
    with open(urdf_file_path, 'r') as infp:
        urdf_content = infp.read()


    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_content}]
    )

    joint_state_publisher_node = Node(
        package = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        name = 'joint_state_publisher_gui',
        output = 'screen'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([mobiman_pkg_dir, 'urdf', 'R1_Lite','urdf', 'mmp_revB_invconfig_upright_a1x.rviz'])]
    )

    
    return LaunchDescription([
        rviz_node,
        joint_state_publisher_node,
        robot_state_publisher_node,
    ])