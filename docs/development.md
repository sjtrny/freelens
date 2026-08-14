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

PyPI publishing is tied to GitHub Releases. The workflow in
`.github/workflows/release.yml` runs when a GitHub Release is published. Saving a draft
does not trigger it.

The workflow runs the checks, verifies that the release tag matches the package version,
and builds the wheel and source distribution. The final job waits for approval in the
`pypi` GitHub environment, then authenticates to PyPI through Trusted Publishing and
uploads both files.

### One-time setup

Add a GitHub publisher in the
[PyPI publishing settings](https://pypi.org/manage/project/freelens/settings/publishing/)
with these values:

- PyPI project: `freelens`
- GitHub owner: `sjtrny`
- Repository: `freelens`
- Workflow: `release.yml`
- Environment: `pypi`

Use the `pypi` GitHub environment with required approval.

### Publish a release

1. Update the version in `pyproject.toml` and merge it to `main`.
1. Open [GitHub Releases](https://github.com/sjtrny/freelens/releases/new).
1. Create a tag named `v<version>`, such as `v0.0.4`, targeting `main`.
1. Publish the release.
1. Open the workflow run and approve the `pypi` deployment.

The release tag must match the version in `pyproject.toml`. The workflow rejects any
mismatch.

The GitHub CLI is an optional alternative to steps 2–4:

```bash
gh release create v0.0.4 --target main --title "FreeLens 0.0.4" --generate-notes
```

## Contributing

Keep changes focused, add tests for changed behavior, update relevant docs, and run the
checks above before opening a pull request.
