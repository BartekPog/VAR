
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="varoptimizer",
    version="0.5.1",
    author="Bartłomiej Pogodziński",
    author_email="bartek.pogod@gmail.com",
    description="Use Visual Autoregressive Models to optimize your images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BartekPog/VAR",
    packages=setuptools.find_packages(exclude=["tests*"]),
    license='MIT',
    python_requires='>=3.11',
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.18.1",
        "Pillow>=9.4.0",
        "huggingface_hub>=0.23",
        "numpy>=1.26.4",
        "pytz>=2022.7",
        "transformers>=4.42",
        "typed-argument-parser>=1.10.1",
    ],
)