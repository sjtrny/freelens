# NaviLens-provided corpus status

The requested validation corpus has been supplied locally, but it is not
redistributed in this repository.

## Provenance and contents

- Dataset description: NaviLens-provided codes, as described by the requester.
- Supplied by: the requester, directly to the remediation workspace.
- Date received: 2026-08-11.
- Delivery method: local workspace file; no public download URL was supplied.
- Original archive: `NaviLens Codes.zip`.
- Archive SHA-256:
  `a93706ebe73b17e37e55af015151c531c396e0d5ccb3bfddaba0156edb33b07a`.
- Contents: 142 original one-page PDFs and 142 unique test cases.
- Archive member timestamps: 2024-12-18 10:04 local ZIP time.
- Transformations: none retained. Validation rasterizes each PDF transiently in
  memory at a 3× scale.
- Redistribution permission: not provided. The archive, source PDFs, and
  rasterized images must remain untracked until authorization is documented.

The corpus has 135 filenames labelled 210 mm, six labelled 105 mm, and one
labelled 50 mm. Every six-hex label has the `AAB` prefix, fixing the upper 12
payload bits, so this is broad evidence across 142 tags but not exhaustive
24-bit payload coverage. The pristine PDFs also do not replace detection tests
using photographs under varied lighting, distance, perspective, or occlusion.

## Local verification

Read and validate the original ZIP directly, without extracting or retaining
any corpus files:

```bash
python -m pip install -e ".[dataset,test]"
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"

FREELENS_NAVILENS_ARCHIVE="/path/to/NaviLens Codes.zip" \
  python -m pytest -q tests/test_navilens_archive.py
```

The direct verifier derives each independent 24-bit expected message from the
six-hex filename label. It does not generate an expected tag with the CRC code
under test, and it requires the exact archive SHA-256 recorded above. On
2026-08-11, the complete supplied archive produced `142/142 cases passed`.

## Future authorized import

If redistribution permission is obtained, import the archive without network
access, review the generated manifest and hashes, and add the binary files
through Git LFS:

```bash
python scripts/import_navilens_dataset.py \
  "/path/to/NaviLens Codes.zip" \
  dataset/navilens-provided
python scripts/verify_navilens_dataset.py \
  --manifest dataset/navilens-provided/manifest.csv
sha256sum --check dataset/navilens-provided/SHA256SUMS
```

The importer uses the same independent filename-derived messages. Until the
permission gate is resolved, do not run this command with a repository
destination for committed output.
