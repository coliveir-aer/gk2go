from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gk2go",
    version="1.1.0",
    author="Craig Oliveira",
    description="A Python library for discovering and loading GK2A satellite data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coliveir-aer/gk2go",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires='>=3.7',
    install_requires=requirements, # Read dependencies from the file
)
