from pathlib import Path

from setuptools import find_packages, setup


def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="lite-pytorch",
    version="0.1.0",
    description="LiTE: Lightweight Inception Time for time series (PyTorch implementation)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Quentin",
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "pandas",
    ],
    packages=find_packages(include=["src", "src.*", "utils", "utils.*"]),
    include_package_data=True,
)
