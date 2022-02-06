from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas"
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="gee",
    version="0.0.1",
    author="Censheng Chen & c",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages = ["gee"],
    install_requires=requirements,
)
