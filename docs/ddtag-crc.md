# CRCs in ddTags

## Patent CRC

The patent defines grids with sizes 5x5, 7x7, 9x9, and 11x11. For a grid with width `N`:

- the centre row and centre column hold the CRC, except for the centre cell;
- the CRC uses `2N - 2` cells and therefore contains `4N - 4` bits;
- the four corners hold the palette;
- the centre cell holds the grid size; and
- all remaining cells form the message.

The message length is `2N^2 - 4N - 6` bits. The patent says to calculate the CRC from
this message only.

### Polynomials and parameters

The patent gives a standard CRC name for each grid size. Its table calls these names
"CRC polynomials", but does not list the initial value, final XOR, or reflection
settings. Interpreting each name as the standard CRC model gives the following
parameters. `check` is the CRC of the ASCII bytes `123456789` and is included to remove
any ambiguity.

All four models use `refin=false` and `refout=false`, so each input byte is processed
most-significant bit first.

| Grid  |  Message |     CRC | Patent name     |         `poly` |         `init` |       `xorout` |        `check` |
| ----- | -------: | ------: | --------------- | -------------: | -------------: | -------------: | -------------: |
| 5x5   |  24 bits | 16 bits | CRC-16-CDMA2000 |       `0xC867` |       `0xFFFF` |       `0x0000` |       `0x4C06` |
| 7x7   |  64 bits | 24 bits | CRC-24-Radix-64 |     `0x864CFB` |     `0xB704CE` |     `0x000000` |     `0x21CF02` |
| 9x9   | 120 bits | 32 bits | CRC-32Q         |   `0x814141AB` |   `0x00000000` |   `0x00000000` |   `0x3010BF7F` |
| 11x11 | 192 bits | 40 bits | CRC-40-GSM      | `0x0004820009` | `0x0000000000` | `0xFFFFFFFFFF` | `0xD4164FC646` |

`CRC-24-Radix-64` is now normally called `CRC-24/OPENPGP`. `CRC-32Q` is also called
`CRC-32/AIXM`. The residue is zero for the 16-, 24-, and 32-bit models. The CRC-40/GSM
residue is `0xC4FF8071FF`.

The hexadecimal `poly` value omits the leading `x^width` term. The complete generator
polynomials are:

```text
CRC-16: x^16 + x^15 + x^14 + x^11 + x^6 + x^5 + x^2 + x + 1
CRC-24: x^24 + x^23 + x^18 + x^17 + x^14 + x^11 + x^10
        + x^7 + x^6 + x^5 + x^4 + x^3 + x + 1
CRC-32: x^32 + x^31 + x^24 + x^22 + x^16 + x^14 + x^8
        + x^7 + x^5 + x^3 + x + 1
CRC-40: x^40 + x^26 + x^23 + x^17 + x^3 + 1
```

### CRC cells

The patent reads the CRC cells in normal matrix order: left to right, then top to
bottom. The following table lists the zero-based cell indices in that order.

| Grid  | CRC cell indices                                                                  |
| ----- | --------------------------------------------------------------------------------- |
| 5x5   | `2, 7, 10, 11, 13, 14, 17, 22`                                                    |
| 7x7   | `3, 10, 17, 21, 22, 23, 25, 26, 27, 31, 38, 45`                                   |
| 9x9   | `4, 13, 22, 31, 36, 37, 38, 39, 41, 42, 43, 44, 49, 58, 67, 76`                   |
| 11x11 | `5, 16, 27, 38, 49, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 71, 82, 93, 104, 115` |

The patent is less exact about the message. It says to compose the message from the
cells that are not palette, CRC, or size cells. It does not give a separate cell order
or say how to pack the resulting bits into bytes. A simple reading is to use the same
row order, but this is not stated as clearly as the CRC cell order.

## Observed CRC

Real 5x5 tags do not use the patent calculation. They use the same `0xC867` polynomial,
but change the input data, initial value, and CRC cell order.

### Message order

The twelve message cells are read by columns:

```text
5, 15,
1, 6, 16, 21,
3, 8, 18, 23,
9, 19
```

Their two-bit values form the 24-bit message.

### CRC input

The CRC covers every cell outside the centre row and centre column. This includes the
four palette corners. The centre cell and the CRC cells are not included.

The cells are read by columns in this order:

```text
0, 5, 15, 20,
1, 6, 16, 21,
3, 8, 18, 23,
4, 9, 19, 24
```

Join the two-bit cell values to make 32 bits. Split those bits from left to right into
four bytes. Keep the input width fixed at four bytes so that leading zero bytes are
preserved.

The CRC parameters are:

```text
width     = 16
poly      = 0xC867
init      = 0x0000
xorout    = 0x0000
refin     = false
refout    = false
check     = 0xE355
residue   = 0x0000
```

This is not the standard CRC-16/CDMA2000 model. That model uses `init=0xFFFF`.

### CRC storage

Write the 16-bit result most-significant bit first and split it into eight two-bit
cells. Store those cells in this order:

```text
10, 11, 2, 7, 17, 22, 13, 14
```

In grid terms, this order is the left arm of the centre row, the upper arm of the centre
column, the lower arm, and then the right arm.

The complete check is equivalent to:

```text
input_bits  = join(cells[i] for i in CRC_INPUT_INDICES)
input_bytes = split input_bits into four 8-bit values, left to right
calculated  = CRC(input_bytes, poly=0xC867, init=0x0000,
                  xorout=0x0000, refin=false, refout=false)
stored      = join(cells[i] for i in CRC_STORAGE_INDICES)
valid       = calculated == integer value of stored
```

### Patent and real 5x5 tags

| Detail                      | Patent 5x5                     | Real NaviLens 5x5                |
| --------------------------- | ------------------------------ | -------------------------------- |
| CRC input                   | 24 message bits                | 32 bits from all non-cross cells |
| Palette corners included    | No                             | Yes                              |
| Polynomial                  | `0xC867`                       | `0xC867`                         |
| Initial value               | `0xFFFF` in the named standard | `0x0000`                         |
| Input and output reflection | False                          | False                            |
| Final XOR                   | `0x0000`                       | `0x0000`                         |
| CRC cell order              | `2, 7, 10, 11, 13, 14, 17, 22` | `10, 11, 2, 7, 17, 22, 13, 14`   |

## Examples

For the Melbourne tram tag:

```text
grid       00101100010110110011111100001000001110001100110010
message    010010100000000010001100
CRC input  00010011101000000000100001110010
bytes      13 A0 08 72
CRC        FFF2
```

Standard CRC-16/CDMA2000 over the message bytes `4A 00 8C` gives `E751`, not `FFF2`.

For the tag labelled `B1269C`:

```text
grid       00001001011001011011001100011111001010001110100110
message    101100010010011010011100
CRC input  00101111000100100110100101110010
bytes      2F 12 69 72
CRC        39A7
```

The [5x5 CRC tests](../tests/test_crc_5x5.py) check these values, the cell orders, byte
packing, tag generation, and changes to each type of cell.

## FreeLens

FreeLens calculates and validates CRCs only for 5x5 tags. It can parse 7x7, 9x9, and
11x11 grids only when CRC validation is disabled:

```python
tag = Tag(bit_string, n=7, validate_crc=False)
assert tag.crc_valid is None
```

The patent describes the CRC layout and names a polynomial for each larger grid.
FreeLens does not implement those CRCs because no real 7x7, 9x9, or 11x11 tags have been
tested.

The local `NaviLens Codes.zip` archive contains 142 5x5 PDF tags. All 142 pass the real
5x5 calculation. The archive is not committed because redistribution permission has not
been provided. Run the check with:

```bash
python -m pip install -e ".[dataset,test]"
python scripts/verify_navilens_archive.py "/path/to/NaviLens Codes.zip"
```

## References

- [ddTag patent: EP 3561729 A1](https://data.epo.org/publication-server/rest/v1.0/publication-dates/20191030/patents/EP3561729NWA1/document.pdf)
- [CRC RevEng catalogue](https://reveng.sourceforge.io/crc-catalogue/)
- [FreeLens 5x5 CRC implementation](../freelens.py)
