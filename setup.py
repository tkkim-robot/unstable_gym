from setuptools import setup

setup(
    name='unstable_gym',
    version='0.1.0',
    packages=['unstable_gym'],
    url='https://github.com/ktk1501/unstable_gym',
    license='',
    author='Taekyung Kim',
    author_email='ktk1501@kakao.com',
    description='A gym-like classical control benchmark for evaluating the robustnesses of control and reinforcement learning algorithms.',
    install_requires=[
        'gym',
        'numpy',
    ],
    tests_require=[
    ]
)
