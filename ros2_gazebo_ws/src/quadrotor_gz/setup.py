from glob import glob
from setuptools import find_packages, setup

package_name = 'quadrotor_gz'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/models/learning_quadrotor',
         glob('models/learning_quadrotor/*')),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='Ashraf Faraj',
    maintainer_email='ashraffaraj1999-wq@users.noreply.github.com',
    description='Full-physics Gazebo validation of PID and PPO quadrotor controllers',
    license='MIT',
    entry_points={'console_scripts': [
        'controller = quadrotor_gz.controller_node:main',
        'recorder = quadrotor_gz.recorder_node:main',
        'performance_overlay = quadrotor_gz.performance_overlay_node:main',
    ]},
)
