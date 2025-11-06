"""Setup script for intraday_system package."""

from setuptools import setup, find_packages

setup(
    name="intraday_system",
    version="1.0.0",
    description="Production-grade intraday trading system for Forex/Metals with Polygon API integration",
    author="Quant Dev Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "pyyaml>=6.0",
        "pyarrow>=12.0.0",  # For parquet support
        "python-dotenv>=1.0.0",  # For .env file support
        "requests>=2.31.0",  # For Polygon API
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ]
    },
    entry_points={
        'console_scripts': [
            'intraday-train=intraday_system.cli.train:main',
        ],
    },
    include_package_data=True,
    package_data={
        'intraday_system': ['config/*.yaml'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
