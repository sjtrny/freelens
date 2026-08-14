# Testing NaviLens code PDFs

Request the free code kit from the
[NaviLens download page](https://www.navilens.com/en/free-codes). Save the received ZIP
file.

## Setup

```bash
source ./setup.sh
uv pip install -e ".[dataset,test]"
```

## Run the verifier

```bash
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

On success, the verifier prints the number of decoded cases. For example:

```text
142 cases passed
```

The verifier reads each PDF from the ZIP, derives the expected 24-bit message from its
filename, renders the PDF, detects the 5×5 tag, and validates its CRC. The reported case
count depends on the downloaded archive.

## Run through pytest

```bash
FREELENS_NAVILENS_ARCHIVE="/path/to/NaviLens Codes.zip" \
  python -m pytest -q tests/test_navilens_archive.py
```

The integration test is skipped when `FREELENS_NAVILENS_ARCHIVE` is unset.
