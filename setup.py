from setuptools import setup

setup(
    name="kfilter",
    version="1.1",
    description="カルマンフィルタを簡易的に扱うためのフレームワーク",
    install_requires=open("requirements.txt").read().splitlines(),
)