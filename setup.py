from setuptools import setup

setup(
    name="kf",
    version="1.0",
    description="カルマンフィルタのテスト実装",
    install_requires=open("requirements.txt").read().splitlines(),
)