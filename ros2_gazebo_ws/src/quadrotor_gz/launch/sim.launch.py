import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('quadrotor_gz')
    world = os.path.join(share, 'worlds', 'quadrotor_world.sdf')
    model_root = os.path.join(share, 'models')
    bridge_config = os.path.join(share, 'config', 'bridge.yaml')
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    checkpoint_default = os.environ.get('QUADROTOR_PPO_CHECKPOINT', '')

    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='pid', description='pid, ppo, hover, or idle'),
        DeclareLaunchArgument('checkpoint', default_value=checkpoint_default),
        DeclareLaunchArgument(
            'start_paused', default_value='false',
            description='Open Gazebo paused so a slow policy can load before physics starts'),
        DeclareLaunchArgument(
            'show_overlay', default_value='true',
            description='Show the always-on-top live tracking-performance plots'),
        DeclareLaunchArgument(
            'overlay_window_seconds', default_value='20.0',
            description='Length of the rolling performance-plot window'),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            model_root + (os.pathsep + existing_resource_path if existing_resource_path else '')),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world],
            condition=UnlessCondition(LaunchConfiguration('start_paused')),
            output='screen'),
        ExecuteProcess(
            cmd=['gz', 'sim', world],
            condition=IfCondition(LaunchConfiguration('start_paused')),
            output='screen'),
        Node(
            package='ros_gz_bridge', executable='parameter_bridge', name='quadrotor_bridge',
            parameters=[{'config_file': bridge_config}], output='screen'),
        Node(
            package='quadrotor_gz', executable='controller', name='quadrotor_controller',
            parameters=[{
                'controller': LaunchConfiguration('controller'),
                'checkpoint': LaunchConfiguration('checkpoint'),
                'use_sim_time': True,
            }], output='screen'),
        Node(
            package='quadrotor_gz', executable='performance_overlay',
            name='quadrotor_performance_overlay',
            condition=IfCondition(LaunchConfiguration('show_overlay')),
            parameters=[{
                'controller': LaunchConfiguration('controller'),
                'window_seconds': ParameterValue(
                    LaunchConfiguration('overlay_window_seconds'), value_type=float),
                'use_sim_time': True,
            }], output='screen'),
    ])
