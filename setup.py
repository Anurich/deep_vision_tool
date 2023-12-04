from setuptools import find_packages, setup
import os

def read_requirements(path):
    fdata = open(path, "r").read()
    return [line.strip() for line in fdata.splitlines()]

def read_from_file(path):
    with open(path, 'r', encoding='utf8') as fin:
        return fin.read()

setup(
    name='vision_tool',
    version='0.0.2',
    description="Make your computer vision task easy",
    url="https://github.com/Anurich/hair_style_recommendation",
    author="AI Team",
    long_description=read_from_file("README.md"),
    long_description_content_type="text/markdown",
    author_email="nautiyalanupam98@gmail.com",
    include_package_data=True,
    packages=find_packages(exclude=["tests", "logs"]),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8",
)
# python -m build
# python3 -m twine upload --repository testpypi dist/*