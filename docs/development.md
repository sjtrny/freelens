# Development

To develop FreeLens, use Python 3.11 or a newer version, `uv`, and Git LFS.

## Setup

Run these commands from the repository root:

```bash
git lfs pull
source ./setup.sh
uv pip install -e ".[test]"
```

The first command downloads files stored in Git LFS. The second command runs `setup.sh` in the current shell. This script creates a Python 3.11 `.venv`, activates it, and installs the runtime requirements. The third command installs the developer tools and FreeLens in editable mode.

## Checks

```bash
python -m pytest -q
python -m black --check freelens.py scripts tests
python -m isort --check-only freelens.py scripts tests
python -m mdformat --check README.md docs dataset
git lfs fsck
```

Continuous integration (CI) checks formatting and imports on Python 3.11 for each pull request. It runs unit tests on Python 3.11 and 3.14. These runs do not include tests marked `integration`.

To apply formatting, run these commands:

```bash
python -m black freelens.py scripts tests
python -m isort freelens.py scripts tests
python -m mdformat README.md docs dataset
```

## Documentation

Use Simplified Technical English with Australian English spelling. Keep technical names, code identifiers, and quoted source data unchanged. Use one source line for each Markdown paragraph. The Markdown formatter preserves this paragraph format.

Use `5×5`, `7×7`, `9×9`, and `11×11` for grid sizes in prose and diagram labels. Use "read positions" for a sequence that starts at 1. Use "cell indices" for grid positions that start at 0.

For a diagram that shows procedure steps, use blue number badges that agree with the numbered instructions. Use 40-pixel circles for single- and double-digit steps. Keep a 12-pixel gap between each badge and its label. Use wider badges for step ranges. Keep these badges separate from cell indices and sample counts. Do not assign a step number to a source image or a layout illustration.

The diagram generators are in `docs/assets/`. They use the Cairo Visuals project and its Atkinson Hyperlegible font. To regenerate the diagrams, set `PYTHONPATH` to your Cairo Visuals checkout. For example:

```bash
PYTHONPATH=/path/to/cairo-visuals python docs/assets/ddtag_crc_diagrams.py
```

Use square corners for tags and cells. Keep figure backgrounds within their borders. Use a 32-pixel content margin around each figure. Use a minimum font size of 18 pixels for bit labels and binary values. Check that all text is readable at its displayed size.

Use the arrows in `image-processing.svg` as the minimum: 38 pixels from tail to tip, with 28 pixels of shaft before the arrowhead. For larger heads or arrows with two heads, keep at least 28 pixels of shaft outside the heads. Increase the space between components to fit the arrows.

## Dataset

Refer to [Dataset](./dataset.md) for the field photograph benchmark. Use the [tag editor](./tag-editor.md) to review labels and edit tag locations.

Optional dataset tests do not run when their files are not available. To check the local PDF archive, run these commands:

```bash
uv pip install -e ".[dataset,test]"
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

Do not commit the archive or its generated files.

## Build

```bash
uv pip install -r requirements_build.txt
python -m flit build --no-use-vcs
```

## Release

The workflow in `.github/workflows/release.yml` runs when you publish a GitHub Release. It does not run when you save a draft.

The workflow checks the code, compares the release tag with the package version, and builds the wheel and source distribution. The last job waits for approval in the `pypi` GitHub environment. It then authenticates to PyPI through Trusted Publishing and uploads the two files.

### One-time setup

Add a GitHub publisher in the [PyPI publishing settings](https://pypi.org/manage/project/freelens/settings/publishing/) with these values:

- PyPI project: `freelens`
- GitHub owner: `sjtrny`
- Repository: `freelens`
- Workflow: `release.yml`
- Environment: `pypi`

Configure the `pypi` GitHub environment to require approval.

### Publish a release

1. Update the version in `pyproject.toml`.
1. Merge the version change to `main`.
1. Open [GitHub Releases](https://github.com/sjtrny/freelens/releases/new).
1. Create a tag named `v<version>`, such as `v0.0.5`, with `main` as its target.
1. Publish the release.
1. Open the workflow run.
1. Approve the `pypi` deployment.

The release tag must agree with the version in `pyproject.toml`. The workflow rejects a mismatch.

The GitHub CLI is an optional alternative to steps 3–5:

```bash
gh release create v0.0.5 --target main --title "FreeLens 0.0.5" --generate-notes
```

## Contributing

1. Limit each change to one task.
1. Add tests for behaviour changes.
1. Update the applicable documentation.
1. Run the commands in the [Checks section](#checks).
1. Open a pull request.
