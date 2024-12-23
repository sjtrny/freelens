*This is a work in progress and sudden, dramatic changes are expected!*

# FreeLens

This project provides a reference implementation of [NaviLens][3] and [ddTag][4] 
for educational or personal use. Commercial use is at your own risk as NaviLens and 
ddTag may attempt to enforce their patents.

## Generating Tags

```
from freelens import Tag

message = "101010101011000000001011"

tag = Tag.from_message(message, 5)

tag_img = tag.to_image()

tag_img.save("tag.png")
```

## Detecting Tags

```
from freelens import detect_tags
from PIL import Image

img = Image.open("dataset/positives/PXL_20241124_081401367.MP.jpg")

tags_list = detect_tags(img, n=5)

for tag in tags_list:
    print(tag.message)
```

## NaviLens

NaviLens is a service that provides navigational data resolution, based on ddTags ("distant dense tag")
placed around the environment. The tags are designed so that they can be quickly scanned
with a mobile device while moving or on moving objects and at a much greater distance than
QR codes. A scanned tag is converted into a message on a users device, the message is then sent 
to the NaviLens web service, which responds with the associated data for the tag.

The main use case of NaviLens is to improve navigation for those with visual impairments. 
For example a tag could be placed on the front of a bus so that a visually impaired person
can hold their phone camera up to the approaching bus to read the route number.

These tag images must be requested from NaviLens who are the central data resolution authority and
maintain a database with the associated data for each ddTag. NaviLens divides the world into smaller geographic 
regions, within which it allocates tags to users. This allows tags to be re-used and overcomes the relatively short 
message length of the tags used by NaviLens. For example the typical 5x5 ddTag can only represent 16,777,216 
unique combinations.

## ddTag Specification

A ddTag consists of a small grid of coloured squares, which represents a "message" of binary data. The typical 
implementation uses a 5x5 grid, which represents a 24 bit message, with cyan, magenta, yellow and black coloured cells.

The data encoded in the tag consists of two parts:
1. the "message", which is a binary sequence, and 
2. a CRC checksum, used to verify the message.

ddTags officially come in the following sizes:
- 5x5
- 7x7
- 9x9
- 11x11

The tag visually consists of three nested components, which are from the outside moving inwards:
1. Outer quiet zone
2. Inner quiet zone
3. Tag grid

### Quiet Zones

The outer quiet zone is a border region of a solid colour, usually white, and should be at least
as thick as the inner quiet zone for best detection results.  The inner quiet zone is a border 
region of a solid colour which must be:
- one on of the four colours used by the ddTag code grid, typically
black,
- the same width as the cells in the tag grid.

### Grid

The grid consists of a square grid, with each cell coloured by one of four colours. Each cell in the grid represents 
two bits of data (`00`, `01`, `10` and `11`) since it is in one of four states.

The grid uses an odd numbered size because an unambiguous center position is required for two features:
- central cell is used to encode the grid size
- central row and column are used to hold a CRC checksum

#### Corners and Colours

The corners do not contain any message data. Instead, they are used as follows:
- The bottom left cell must contain the darkest colour, e.g. black, from the colour palette as this is used to orient 
the tag.
- The other corners are used to infer the colour palette used by the tag, so they must have distinct colours.
- The other corners determine bit value associated with each colour, which starting from the top left and moving 
  clockwise around the grid are `00`, `01`, `10`, `11`. For example if the top left corner is cyan then all cyan cells 
  have the value `00`.

#### Center Cell

The central cell does not contain any message data. In the patent, this cell is reserved 
for encoding the size of the grid. The patent uses the following encoding scheme:

| NxN     | Center Cell |
|---------|-------------|
| 5x5     | cyan        |
| 7x7     | magenta     |
| 9x9     | yellow      |
| 11 X 11 | black       |

#### CRC

To ensure data integrity, the grid contains a CRC code, which is contained in the center row and column
excluding the center cell. The CRC length varies with the dimensions of the grid. The patent lists the following
correspondence:

| NxN     | Message length | CRC length | CRC Polynomial  |
|---------|----------------|------------|-----------------|
| 5x5     | 24 bits        | 16 bits    | CRC-16-CDMA2000 |
| 7x7     | 64 bits        | 24 bits    | CRC-24-Radix-64 |
| 9x9     | 120 bits       | 32 bits    | CRC-32Q         |
| 11 X 11 | 192 bits       | 40 bits    | CRC-40-GSM      |

*Note that I have not been able to verify the checksums on NaviLens generated tags [as described in this issue](https://github.com/sjtrny/freelens/issues/1).*

*The patent does not describe the process for generating the CRC apart from "The CRC is calculated from the message"
and it is possible that the message is transformed before the CRC is generated.*

*This library generates CRCs from the message content without any prior transformation.*

#### Message

The message is formed by concatenating the binary values of the remaining cells in "reading order",
which is described in the patent as:

> from left to right and from top to bottom

Most sane people would interpret this as reading row by row, starting with the first row, reading all the values in it 
from left to right, and then moving to the next row below it. However, this interpretation is incorrect for tags
distributed by NaviLens, which are read column by column.

To maximise compatibility, we have adopted this psychotic interpretation.

## Detecting Possible Tags

The ddTag patent does not describe a particular process for detecting "frames" and leaves
it up to the implementor. However it suggests that \[2\] may be used.

In `detect_frames` we use a modified version of \[2\] as follows:
1. Convert image to grayscale
2. Detect edges by local adaptive thresholding
3. Detect contours by Suzuki's method
4. Fit polygon to contours
5. Apply filters:
   1. 4-vertex polygons.
   2. Area greater than threshold
   3. Convex polygon
   4. Shape is roughly square (perimeter/area test)
   5. Check that border around frame is white

## Decoding Possible Tags

This process is adapted from the patent.

For each un-rectified frame polygon:
1. Convert image to CIELab colour space
2. Un-warp frame image to square aspect ratio and resize to a fixed size
3. Get the cell colours from the center positions of each cell in the grid
4. Obtain the palette colours from the four corners of the grid 
5. Assign each cell in the grid to the closest colour in the palette
6. Validate tag
   1. Convert cells to binary using the rule that the palette is ordered clockwise starting at the top left with the binary values `00`, `01`, `10`, `11`. 
   2. Extract message code and CRC code.
   3. Validate message code with CRC code.

## References

\[1\]: European Patent [EP3561729NWA1][1]

\[2\]: Garrido-Jurado, S., et al. (2014). [Automatic generation and detection of highly reliable fiducial markers under occlusion][2]. Pattern Recognition.


[1]: https://data.epo.org/publication-server/rest/v1.0/publication-dates/20191030/patents/EP3561729NWA1/document.pdf
[2]: https://cs-courses.mines.edu/csci507/schedule/24/ArUco.pdf
[3]: https://www.navilens.com
[4]: https://www.ddtags.com
