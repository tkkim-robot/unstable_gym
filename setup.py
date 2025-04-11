from setuptools import setup, find_packages

setup(
    name='unstable_gym',
    version='0.1.0',
    packages=find_packages(),
    author='Taekyung Kim',
    author_email='ktk1501@kakao.com',
    description='A gym-like classical control benchmark for evaluating the robustnesses of control and reinforcement learning algorithms.',
    install_requires=[
        'gymnasium',
        'numpy',
    ],
)
