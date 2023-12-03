from setuptools import find_packages, setup
import os

def read_requirements(path):
    fdata = open(path, "r").read()
    return [line.strip() for line in fdata.splitlines()]


setup(
    name='deep_vision_tool',
    version='0.1.4',
    description="Make your computer vision task easy",
    url="https://github.com/Anurich/hair_style_recommendation",
    author="AI Team",
    author_email="nautiyalanupam98@gmail.com",
    include_package_data=True,
    packages=find_packages(exclude=["tests", "logs"]),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8",
)