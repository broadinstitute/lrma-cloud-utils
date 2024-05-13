from io import open

from setuptools import find_packages, setup

with open('requirements.txt', 'r') as fh:
    to_be_installed = [l.rstrip('\n') for l in fh.readlines()]

with open('test-requirements.txt', 'r') as fh:
    test_dev_install = [l.rstrip('\n') for l in fh.readlines()]

version = "0.0.10"
setup(
    name="lrmaCUX",
    version=version,
    description="A python library for interacting with GCS, Cromwell, and Terra",
    url="https://github.com/broadinstitute/lrma-cloud-utils",
    author="Steve Huang",
    author_email="shuang@broadinstitute.org",
    license="BSD 3-Clause",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    python_requires=">=3.7",
    install_requires=to_be_installed,
    tests_require=test_dev_install,

    packages=find_packages(
        where="src",
        include=['lrmaCU*', 'smrtlink_parser*']
    ),
    package_dir={"": "src"},
    include_package_data=True,

    classifiers=[
        "Development Status :: 1 - preAlpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 3-Clause",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
