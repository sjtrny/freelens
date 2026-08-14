# Development

FreeLens requires Python 3.11+, `uv`, and Git LFS.

## Setup

```bash
git lfs pull
source ./setup.sh
uv pip install -e ".[test]"
```

`setup.sh` creates a Python 3.11 `.venv` and installs the runtime requirements.
Source it to keep the environment active; the second command adds developer
tools and installs FreeLens in editable mode.

## Checks

```bash
python -m pytest -q
python -m black --check freelens.py scripts tests
python -m isort --check-only freelens.py scripts tests
git lfs fsck
```

CI runs these checks on Python 3.11 and 3.14.

To apply Python formatting:

```bash
python -m black freelens.py scripts tests
python -m isort freelens.py scripts tests
```

## Dataset

Optional dataset tests skip when their files are absent. To validate the local
PDF archive:

```bash
uv pip install -e ".[dataset,test]"
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

Do not commit that archive or its generated files.

## Build

```bash
uv pip install -r requirements_build.txt
python -m flit build --no-use-vcs
```

## Contributing

Keep changes focused, add tests for changed behavior, update relevant docs, and
run the checks above before opening a pull request.
