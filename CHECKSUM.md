# The 5x5 ddTag checksum: documented vs deployed

The ddTag patent specifies a CRC for each grid size, and `freelens.valid_crc`
implements it faithfully (for 5x5: CRC-16-CDMA2000, polynomial `0xC867`,
init `0xFFFF`, computed over the message with no pre-transform). The README and
[issue #1](https://github.com/sjtrny/freelens/issues/1) already note the catch:
**that checksum does not match the tags NaviLens actually deploys.**

`realcrc.py` implements the checksum the deployed 5x5 tags really carry. It can
both **generate** and **check** deployed checksums.

## The gap, on a real tag

For the Melbourne-tram tag in `dataset/positives` (the one from issue #1):

| | value |
|---|---|
| Grid (row-major, C=0 M=1 Y=2 K=3) | `[0 2 3 0 1; 1 2 3 0 3; 3 3 0 0 2; 0 0 3 2 0; 3 0 3 0 2]` |
| CRC cells on the tag (`get_crc_inds` order) | `0xfff2` |
| Documented CRC-16-CDMA2000 of the message | `0x1ec4` |
| Deployed checksum (`realcrc`) of the message | `0xfff2` ✓ |

(The original issue wrote the tag's CRC as `0xff2f`; that is the same eight
cells serialized in a different order. Under freelens' current `get_crc_inds`
they read `0xfff2`.)

## The deployed checksum

It is an affine map over GF(2), expressed in freelens' own conventions
(`ind_bit_map` two bits MSB-first per cell, `get_message_inds` / `get_crc_inds`
ordering):

```
crc_bits = (A_f @ message_bits) XOR b_f          (mod 2)
```

`A_f` is 16x24, `b_f` is length 16; both are in `realcrc.py`. Rows are the crc
bits in `get_crc_inds(5)` order; columns are the message bits in
`get_message_inds(5)` order. Equivalently it is a 12-tap transducer: start from
the constant state `b_f` and XOR one contribution per message cell.

Verified against two independent real tags (both reproduce bit-for-bit):

| Tag | Source | Deployed crc |
|---|---|---|
| `B1269C` | printed label | `0x39a7` |
| issue #1 | Melbourne tram photo | `0xfff2` |

## What it is not (open TODO)

The deployed checksum is **not** a closed-form function as far as anyone has
found:

- **Not a CRC-16.** An exhaustive sweep of all 32768 degree-16 generator
  polynomials (built each [40,24] CRC's weight spectrum via dual + MacWilliams)
  finds no match. A structural tell: standard even-distance CRCs carry the factor
  (x+1) and have all-even weights; this code has odd-weight codewords.
- **Not GF(4)-linear** under a per-symbol basis change.
- **No low-dimensional compression**: the 12 message-cell column blocks are
  linearly independent.

As a binary code it is `[40,24]` with minimum distance 5; as cells it is `[20,12]`
over the 4-symbol alphabet with minimum symbol distance 4 (corrects 1 cell error,
detects 3). The evidence points to a computer-searched distance code rather than
an algebraic family, so there may be no prettier form. **Until one is found, the
matrix / transducer in `realcrc.py` is the canonical description.**

Still open: the design *metric*. If the distance optimization were re-run under a
CMYK confusion-weighted metric (how often a reader confuses, say, Y vs K under
blur), the code might become clean under a specific weighting -- that weighting
would be the design principle. This needs the reader's empirical cell-confusion
rates, which are not yet available.

## Scope

Only **n=5** is solved. The 7x7 / 9x9 / 11x11 deployed checksums are unknown
(deriving them needs real tags of those sizes); for those sizes use the
documented `valid_crc`.

## Usage

```python
import realcrc

# check a decoded tag (full 50-char grid bit string, as freelens.Tag uses)
realcrc.valid_real_crc(bit_string)            # -> True / False

# generate a tag that carries the deployed checksum
message = "101010101011000000001011"          # 24 bits
tag_bits = realcrc.apply_real_crc(message)     # full grid bit string
```

In the detector, `validate_crc` selects the scheme (`Tag`, `decode_frames`,
`detect_tags`):

```python
import freelens

tags = freelens.detect_tags(img, n=5, validate_crc="affine")  # deployed checksum
# "patent" -> documented CRC-16 ; None -> no check. tag.valid holds the result.
```

## Reproducing this

```
python scripts/reproduce_claims.py        # affine crc matches both real tags
python scripts/derive_affine_checksum.py  # re-derive A_f/b_f from (msg, crc) pairs
python scripts/validate_dataset.py [dir]   # decode a folder and report validity
```

`derive_affine_checksum.py` shows the generation: given >=25 independent
(message, crc) pairs it solves for `A_f`/`b_f` over GF(2); its self-test recovers
the shipped matrix from random samples, and pointed at real tags it re-derives
the same map.

To check the affine checksum across a whole tag set (e.g. the NaviLens free kit),
drop the images in `dataset/freekit/` or set `FREELENS_FREEKIT=<dir>`;
`tests/test_freekit.py` decodes each and asserts it passes (skips when empty).

## Provenance

The reverse-engineering of the deployed checksum -- the affine map and the proof
that it is not a CRC -- and the decoder design were done without AI assistance.
AI (Claude) assisted in this fork with porting that result into freelens'
conventions, cross-checking it against sample tags, and the git workflow.
