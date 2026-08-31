## ddTag Specification

A ddTag consists of a small grid of coloured squares, which represents a "message" of
binary data. The typical implementation uses a 5x5 grid, which represents a 24 bit
message, with cyan, magenta, yellow and black coloured cells.

![Example 5x5 ddTag encoding message 4A005C](./assets/ddtag/example.svg)

The data encoded in the tag consists of two parts:

1. the "message", which is a binary sequence, and
1. a CRC checksum, used to verify the message.

ddTags officially come in the following sizes:

- 5x5
- 7x7
- 9x9
- 11x11

The tag visually consists of three nested components, which are from the outside moving
inwards:

1. Outer quiet zone
1. Inner quiet zone
1. Tag grid

### Quiet Zones

The outer quiet zone is a border region of a solid colour, usually white, and should be
at least as thick as the inner quiet zone for best detection results. The inner quiet
zone is a border region of a solid colour which must be:

- one on of the four colours used by the ddTag code grid, typically black,
- the same width as the cells in the tag grid.

![Nested outer quiet zone, inner quiet zone, and tag grid](./assets/ddtag/quiet-zones.svg)

### Grid

The grid consists of a square grid, with each cell coloured by one of four colours. Each
cell in the grid represents two bits of data (`00`, `01`, `10` and `11`) since it is in
one of four states.

The grid uses an odd numbered size because an unambiguous center position is required
for two features:

- central cell is used to encode the grid size
- central row and column are used to hold a CRC checksum

#### Corners and Colours

The corners do not contain any message data. Instead, they are used as follows:

- The bottom left cell must contain the darkest colour, e.g. black, from the colour
  palette as this is used to orient the tag.
- The other corners are used to infer the colour palette used by the tag, so they must
  have distinct colours.
- The other corners determine bit value associated with each colour, which starting from
  the top left and moving clockwise around the grid are `00`, `01`, `10`, `11`. For
  example if the top left corner is cyan then all cyan cells have the value `00`.

![Corner cells establish orientation and map palette colours to bit values](./assets/ddtag/corners.svg)

#### Center Cell

The central cell does not contain any message data. In the patent, this cell is reserved
for encoding the size of the grid. The patent uses the following encoding scheme:

| NxN     | Center Cell |
| ------- | ----------- |
| 5x5     | cyan        |
| 7x7     | magenta     |
| 9x9     | yellow      |
| 11 X 11 | black       |

#### CRC

To ensure data integrity, each ddTag reserves certain cells for a CRC. The patent
describes this as using the cells in the central "cross" of the tag, excluding the
center cell. However, deployed ddTags do not appear to conform to the patent. Instead
they also include the corner cells.

For more information about CRC refer to [CRCs in ddTags](./ddtag-crc.md).

![5x5 CRC input cells, stored CRC cross, included palette corners, and size cell](./assets/ddtag/crc.svg)

#### Message

The message is formed by concatenating the binary values of the remaining cells in
"reading order", which is described in the patent as:

> from left to right and from top to bottom

Most sane people would interpret this as reading row by row, starting with the first
row, reading all the values in it from left to right, and then moving to the next row
below it. However, this interpretation is incorrect for tags distributed by NaviLens,
which are read column by column.

To maximise compatibility, we have adopted this psychotic interpretation.

![Message cells numbered in column-major reading order and concatenated into 24 bits](./assets/ddtag/message-order.svg)

## References

European Patent [EP3561729NWA1][1]

[1]: https://data.epo.org/publication-server/rest/v1.0/publication-dates/20191030/patents/EP3561729NWA1/document.pdf
