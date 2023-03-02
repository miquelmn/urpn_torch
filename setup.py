""" Configuration file for pypi package."""
from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name="urpn_torch",
    version="0.1",
    description="U RPN Torch",
    author="Miquel Miró Nicolau",
    author_email="miquel.miro@uib.cat",
    license="MIT",
    packages=["urpn_torch"],
    keywords=["Instance Segmentation", "Deep Learning", "Computer Vision"],
    install_requires=install_requires,
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
