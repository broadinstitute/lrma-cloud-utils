# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

lrmaCU is a Python library for interacting with Google Cloud Storage (GCS), Cromwell, and Terra platform for genomics research. The library provides utilities for managing Terra workspaces, submitting workflows, handling GCS operations, sending email notifications, and diagnosing failed Cromwell jobs.

## Development Environment Setup

```bash
python3 -mvenv venv
. venv/bin/activate
pip install -r dev-requirements.txt
pip install -e .
```

## Essential Commands

### Testing
- `tox` - Run all linting and tests
- `tox -e unit` - Run only unit tests (fast)
- `pytest tests/unit tests/acceptance` - Run specific test directories

### Linting and Code Quality
- `tox -e lint` - Run all linting commands
- `black src tests setup.py` - Reformat all project files
- `isort -rc src tests setup.py` - Sort imports
- `flake8 src tests setup.py` - Check PEP8 compliance
- `pylint src` - Lint Python code for errors and style issues

### Version Management
- Use `bumpversion PART` where PART is `major`, `minor`, or `patch`
- **Never manually edit version numbers**

## Code Architecture

### Core Structure
- `src/lrmaCU/` - Main library code
  - `utils.py` - General utilities including email notifications via SendGrid, FISS API retry logic, and file operations
  - `gcs_utils.py` - Google Cloud Storage operations with `GcsPath` class for blob management
  - `log.py` - Logging configuration
  - `terra/` - Terra platform integration
    - `workspace_utils.py` - Workspace management, bucket operations, cost calculation
    - `table_utils.py` - Data table operations for Terra entities
    - `submission/submission_utils.py` - Workflow submission and configuration management
    - `specialized_utils.py` - Terra-specific utilities
    - `expt_design/` - Experimental design and table upload utilities
  - `cromwell/utils.py` - Cromwell workflow execution analysis and timing diagnostics

### Key Design Patterns
- Uses `retry_fiss_api_call()` wrapper for robust FireCloud API interactions with automatic retry on connection errors
- Terra entities are managed through pandas DataFrames with specific handling for list-type attributes
- GCS operations abstracted through `GcsPath` class providing file/directory-like operations on cloud storage
- Email notifications use SendGrid with support for attachments (text, CSV, PDF)

### Dependencies
- Core: google-cloud libraries (storage, bigquery, monitoring), firecloud, pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Utilities: tqdm, python-dateutil, sendgrid, termcolor

### Testing Structure
- `tests/unit/` - Unit tests
- `tests/acceptance/` - Acceptance tests  
- `tests/integration/` - Integration tests
- Uses pytest with coverage reporting

## Environment Variables
The library expects these environment variables for full functionality:
- `SENDGRID_API_KEY` - For email notifications
- `SENDER_EMAIL` - Default sender email address
- `TZ` - Timezone (defaults to UTC)