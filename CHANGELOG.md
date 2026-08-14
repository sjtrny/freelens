# Changelog

## Unreleased

- Removed the patent-derived CRC implementation and CRC configurations for
  unverified larger tag sizes.
- CRC validation and generation now implement the observed deployed 5×5
  calculation, including all four palette corners.
- Added `Tag.crc_valid`, `Tag.center_valid`, and `Tag.corners_valid` so checksum
  validity is not confused with structural validity. `Tag.valid` remains a
  deprecated alias for `Tag.crc_valid`.
- CRC validation and generation for 7×7, 9×9, and 11×11 tags now raise clear
  errors. Parsing those grids remains available with `validate_crc=False`.
- Added `require_valid_crc` filtering to `decode_frames` and `detect_tags`.
- Added an in-memory verifier for the locally supplied 142-PDF NaviLens archive;
  expected payloads are derived independently from the six-hex filename labels.
