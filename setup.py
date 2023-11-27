from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with requirements_path.open() as file:
        return [line.strip() for line in file if line.strip()]


setup(
    name="colour_som",
    version="1.0.0",
    install_requires=read_requirements(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
