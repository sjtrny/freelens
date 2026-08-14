# Development

FreeLens requires Python 3.11+, `uv`, and Git LFS.

## Setup

```bash
git lfs pull
source ./setup.sh
uv pip install -e ".[test]"
```

`setup.sh` creates a Python 3.11 `.venv` and installs the runtime requirements. Source
it to keep the environment active; the second command adds developer tools and installs
FreeLens in editable mode.

## Checks

```bash
python -m pytest -q
python -m black --check freelens.py scripts tests
python -m isort --check-only freelens.py scripts tests
python -m mdformat --check README.md docs dataset
git lfs fsck
```

CI runs these checks on Python 3.11 and 3.14.

To apply formatting:

```bash
python -m black freelens.py scripts tests
python -m isort freelens.py scripts tests
python -m mdformat README.md docs dataset
```

## Dataset

Optional dataset tests skip when their files are absent. To validate the local PDF
archive:

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

## Release

Add a GitHub publisher in the
[PyPI publishing settings](https://pypi.org/manage/project/freelens/settings/publishing/)
with these values:

- PyPI project: `freelens`
- GitHub owner: `sjtrny`
- Repository: `freelens`
- Workflow: `release.yml`
- Environment: `pypi`

Use the `pypi` GitHub environment with required approval. For each release, update the
version in `pyproject.toml`, merge it to `main`, then publish a GitHub Release with a
matching `v` tag:

```bash
gh release create v0.0.3 --target main --title "FreeLens 0.0.3" --generate-notes
```

The release workflow runs the checks, builds both distributions, and publishes them to
PyPI. It rejects a tag that does not match the package version.

## Contributing

Keep changes focused, add tests for changed behavior, update relevant docs, and run the
checks above before opening a pull request.
