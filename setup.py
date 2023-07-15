from setuptools import setup

setup(
    name="gym_balletenv",
    version="0.0.2",
    install_requires=[
        "gymnasium==0.27.1",
        "pycolab @ git+https://github.com/deepmind/pycolab.git",
        "absl-py",
        "opencv-python",
        "moviepy"
        ],
)