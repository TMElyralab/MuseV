#!/usr/bin/env python

from setuptools import setup, find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
requirements = [x for x in requirements if x and not x.startswith("#")]
requirements = [x.split(" ")[0] if "index-url" in x else x for x in requirements]

setup(
    name="musev",  # used in pip install
    version="1.0.0",
    author="anchorxia, zkangchen",
    author_email="anchorxia@tencent.com, zkangchen@tencent.com",
    description="Package about human video creation",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=find_packages("musev"),
    package_dir={"": "musev"},
    url="https://github.com/TMElyralab/MuseV",
    # include_package_data=True, # please edit MANIFEST.in
    # packages=find_packages(),  # used in import
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # extras_require=extras_require,
    install_requires=requirements,
    dependency_links=["https://download.pytorch.org/whl/cu118"],
)
