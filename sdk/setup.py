#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README file
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version
version = {}
with open(os.path.join(current_dir, "belfry_labs", "__version__.py")) as fp:
    exec(fp.read(), version)

setup(
    name="belfry-labs-sdk",
    version=version["__version__"],
    author="Belfry Labs Team",
    author_email="support@belfrylabs.com",
    description="Official Python SDK for Belfry Labs - Enterprise AI Safety Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/belfry-labs/python-sdk",
    project_urls={
        "Documentation": "https://docs.belfrylabs.com",
        "Source": "https://github.com/belfry-labs/python-sdk",
        "Tracker": "https://github.com/belfry-labs/python-sdk/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "tenacity>=8.0.0",
        "python-dateutil>=2.8.0",
        "rich>=13.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "coverage>=7.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "cli": [
            "rich-cli>=1.8.0",
            "typer>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "belfry-labs=belfry_labs.cli:main",
            "belfry=belfry_labs.cli:main",
            "belfry-mcp=belfry_labs.mcp_server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ai-safety",
        "model-evaluation", 
        "red-team-testing",
        "ai-security",
        "machine-learning",
        "artificial-intelligence",
        "safety-testing",
        "model-benchmarking",
    ],
)