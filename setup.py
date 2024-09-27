from setuptools import find_packages, setup

setup(
    name="metaworld2gym",
    version="1.0.0",
    author="Zibin Dong",
    description=("Gym wrapper and automatic expert demonstration collection for Meta-World"),
    packages=find_packages(),
    install_requires=[
        "gym>=0.15.4",
        "mujoco-py<2.2,>=2.0",
        "numpy>=1.18",
        "torch",
        "termcolor",
        "zarr",
        "numba",
    ],
)
