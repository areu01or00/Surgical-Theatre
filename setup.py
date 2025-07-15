"""Setup configuration for surgical_theater package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surgical-theater",
    version="0.1.0",
    author="SurgicalTheater Contributors",
    author_email="",
    description="Zero-copy model validation during training - Test your models without breaking the bank (or your GPU)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/areu01or00/surgical-theater",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "transformers>=4.0.0",
            "tqdm",
        ],
    },
    keywords="pytorch machine-learning deep-learning model-validation memory-optimization",
    project_urls={
        "Bug Reports": "https://github.com/areu01or00/surgical-theater/issues",
        "Source": "https://github.com/areu01or00/surgical-theater",
    },
)