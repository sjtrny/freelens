
CRC validation and generation implement the behavior observed in deployed 5×5
NaviLens tags. CRC behavior for 7×7, 9×9, and 11×11 tags is not implemented
because it has not been independently verified. Larger grids can still be
parsed explicitly without CRC validation:

```python
tag = Tag(bit_string, n=7, validate_crc=False)
assert tag.crc_valid is None
```

See [CRCs in ddTags](./docs/ddtag-crc.md) for the checksum's place in the tag,
the observed 5×5 algorithm, known-answer fixtures, and local archive validation.


# CRCs in ddTags

A ddTag carries a payload and a cyclic redundancy check (CRC) in its coloured
grid. Each cell has one of four palette values, so it represents two bits. The
CRC lets a reader reject accidental colour or cell errors.

## The deployed 5×5 format

The independently verified format is the 5×5 tag used by NaviLens. Its cells
have four roles:

```text
P D C D P
D D C D D
C C S C C
D D C D D
P D C D P
```

`P` marks a palette corner, `D` a payload cell, `C` a CRC cell, and `S` the
center size marker. The palette corners, clockwise from the top left, represent
`00`, `01`, `10`, and `11`. The 5×5 center marker is `00`.

Numbering the cells row by row gives:

```text
 0  1  2  3  4
 5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24
```

### Payload

The twelve payload cells are read by columns, not rows:

```text
5, 15,
1, 6, 16, 21,
3, 8, 18, 23,
9, 19
```

Concatenating their two-bit values produces the 24-bit message.

### CRC input

The checksum covers every cell outside the center row and column. Unlike the
payload, this includes the four palette corners:

```text
0, 5, 15, 20,
1, 6, 16, 21,
3, 8, 18, 23,
4, 9, 19, 24
```

Read the two-bit values in that order to make a 32-bit string, then split it
left-to-right into four bytes. This explicit split preserves leading zero
bytes. The bytes are processed MSB-first with:

```text
width       = 16
polynomial  = 0xC867
initial     = 0x0000
final XOR   = 0x0000
refin       = false
refout      = false
```

This is sometimes called CRC-16/CDMA2000, but the usual named configuration has
a different initial value. The complete tuple above is what matters.

### CRC storage

The resulting 16 bits are split into eight two-bit values and stored in this
order:

```text
10, 11, 2, 7, 17, 22, 13, 14
```

The center cell is not part of the CRC. FreeLens reports the checksum, center,
and palette layout separately as `crc_valid`, `center_valid`, and
`corners_valid`.

An earlier implementation checksummed only the payload described by the patent.
That calculation does not validate deployed tags. Including the palette corners
and using the zero initial value does.

## Examples from deployed tags

The Melbourne tram fixture from issue #1:

```text
grid     00101100010110110011111100001000001110001100110010
message  010010100000000010001100
input    00010011101000000000100001110010
bytes    13 A0 08 72
CRC      FFF2
```

The tag printed with the label `B1269C`:

```text
grid     00001001011001011011001100011111001010001110100110
message  101100010010011010011100
input    00101111000100100110100101110010
bytes    2F 12 69 72
CRC      39A7
```

The [CRC tests](../tests/test_crc_5x5.py) check both grids, cell order, byte
packing, calculated CRCs, generation, and corruption of each cell role.

## Larger tags

ddTags also use 7×7, 9×9, and 11×11 grids, but their deployed CRC details have
not been independently verified. A generator that validates its own output is
not evidence of compatibility. Each size needs real fixtures that establish its
cell membership, order, packing, parameters, and CRC placement.

FreeLens therefore validates and generates CRCs only for 5×5 tags. Larger grids
can still be parsed with `validate_crc=False`.

## Using FreeLens

```python
tag = Tag(bit_string, n=5, validate_crc=True)
assert tag.crc_valid is True

tags = detect_tags(
    image,
    n=5,
    validate_crc=True,
    require_valid_crc=True,
)
```

For a larger grid:

```python
tag = Tag(bit_string, n=7, validate_crc=False)
assert tag.crc_valid is None
```

CRC validation or generation with `n=7`, `n=9`, or `n=11` raises a
`ValueError`.

## Fixtures

The locally supplied `NaviLens Codes.zip` contains 142 one-page PDFs and has
SHA-256
`a93706ebe73b17e37e55af015151c531c396e0d5ccb3bfddaba0156edb33b07a`.
The [archive verifier](../scripts/verify_navilens_archive.py) reads the PDFs in
memory and derives each expected payload from its six-hex filename:

```bash
python -m pip install -e ".[dataset,test]"
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

All 142 PDFs passed at the default 3× render scale. Their labels share the
`AAB` prefix, so the upper 12 payload bits are constant. The archive remains
uncommitted because redistribution permission was not provided.

The CC-BY-4.0 community photographs in `dataset/positives/` and
`dataset/negatives/` are not yet covered by an automated test. See the
[dataset notes](../dataset/navilens-provided/README.md) for provenance and the
future Git LFS import process.
