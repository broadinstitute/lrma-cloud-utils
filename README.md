# lrmaCU 
A python library for interacting with GCS, Cromwell, and Terra.

The repo is organized into two main components, `src` and `notebooks`.

`src` holds code for the core library, named `lrmaCU`, providing functions for
  * setting up and maintaining Terra workspace tables,
  * submit workflows on Terra workspaces,
  * interacting with GCS,
  * sending out email notifications,
  * diagnosing failed [cromshell](https://github.com/broadinstitute/cromshell/tree/cromshell_2.0) jobs.

`notebooks` contains example codes for using the library.

## Installation

Two options:

  * clone this repo and `pip install .`
  * install from git directly, optionally specifying a commit/branch/tag

    ```bash
    python3 -m pip install --upgrade \
      git+https://github.com/broadinstitute/lrma-cloud-utils.git
    # or
    export commit_hash='...'
    python3 -m pip install --upgrade \
      git+https://github.com/broadinstitute/lrma-cloud-utils@${commit_hash}
    ```

When you add runtime dependencies, add them to `requirements.txt`.

## Development

To do development in this codebase, the python3 development package must
be installed.

After installation the development environment can be set up by
the following commands:

    python3 -mvenv venv
    . venv/bin/activate
    pip install -r dev-requirements.txt
    pip install -e .

When you add development dependencies, add them to `dev-requirements.txt`.

### Linting files

    # run all linting commands
    tox -e lint

    # reformat all project files
    black src tests setup.py

    # sort imports in project files
    isort -rc src tests setup.py

    # check pep8 against all project files
    flake8 src tests setup.py

    # lint python code for common errors and codestyle issues
    pylint src

### Tests

    # run all linting and test
    tox

    # run only (fast) unit tests
    tox -e unit

    # run only linting
    tox -e lint

Note: If you run into "module not found" errors when running tox for testing, verify the modules are listed in `test-requirements.txt` and delete the .tox folder to force tox to refresh dependencies.

When running `tox`, you'll notice that the linter runs before black8.  This is intentional.  Rather than have it blindly reformat your code, I wanted to make sure you knew what you were doing wrong (i.e. against PEP 8 standards).  You can configure this order to suit your needs.

When you add testing dependencies, add them to `test-requirements.txt`.

### Versioning

We use `bumpversion` to maintain version numbers.
***DO NOT MANUALLY EDIT ANY VERSION NUMBERS.***

Our versions are specified by a 3 number semantic version system (https://semver.org/):

	major.minor.patch

To update the version with bumpversion do the following:

`bumpversion PART` where PART is one of:
- major
- minor
- patch

This will increase the corresponding version number by 1.

