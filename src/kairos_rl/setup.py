from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kairos_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'gymnasium',
        'stable-baselines3',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='milka',
    maintainer_email='milka@todo.todo',
    description='Reinforcement Learning environment for KAIROS Pick & Place',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_ppo = kairos_rl.train_ppo:main',
            'train_ppo_parallel = kairos_rl.train_ppo_parallel:main',
            'train_curriculum = kairos_rl.train_curriculum:main',
            'eval_curriculum = kairos_rl.eval_curriculum:main',
            'eval_policy = kairos_rl.eval_curriculum:main',
        ],
    },
)
