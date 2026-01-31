#!/usr/bin/env python3
"""
Setup script for the refactored_generator_suite package.
"""

from setuptools import setup, find_packages

setup(
    name="refactored_generator_suite",
    version="0.1.0",
    description="A comprehensive framework for generating standardized test files for HuggingFace models",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/your-repo",
    packages=find_packages(),
    package_data={
        "refactored_generator_suite": ["configs/*.yaml", "docs/*.md"],
    },
    entry_points={
        "console_scripts": [
            "generate-test=refactored_generator_suite.scripts.generate_test:main",
            "validate-template=refactored_generator_suite.scripts.validate_template:main",
            "benchmark-generator=refactored_generator_suite.scripts.benchmark_generator:main",
        ],
    },
    install_requires=[
        "pyyaml>=6.0",
        "jinja2>=3.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "tox>=3.24.0",
            "sphinx>=4.5.0",
            "pre-commit>=2.17.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)