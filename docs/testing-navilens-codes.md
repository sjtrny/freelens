# Testing NaviLens code PDFs

1. Request the free code kit from the [NaviLens download page](https://www.navilens.com/en/free-codes).
1. Save the ZIP file that NaviLens sends you.

## Setup

```bash
source ./setup.sh
uv pip install -e ".[dataset,test]"
```

## Run the verifier

```bash
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

If all cases pass, the verifier prints the number of decoded cases. For example:

```text
142 cases passed
```

The verifier processes each PDF in the ZIP file. It gets the expected 24-bit message from the filename. It then renders the PDF, detects the 5×5 tag, and checks its CRC. The number of cases depends on the archive you download.

## Run through pytest

```bash
FREELENS_NAVILENS_ARCHIVE="/path/to/NaviLens Codes.zip" \
  python -m pytest -q tests/test_navilens_archive.py
```

The integration test does not run if `FREELENS_NAVILENS_ARCHIVE` is unset.
