*This is a work in progress and sudden, dramatic changes are expected!*

# FreeLens

This project provides a reference implementation of [NaviLens][3] and [ddTag][4] 
for educational or personal use. Commercial use is at your own risk as NaviLens and 
ddTag may attempt to enforce their patents.

## Components

- ddTag Generator
- ddTag Detector and Decoder

## NaviLens

NaviLens is a service that provides navigational data resolution, based on ddTags
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

NaviLens uses a tag format called "ddTag", short for "distant dense tag". The "ddTag" brand 
is managed by a separate entity to NaviLens, however both companies are closely linked. A ddTag 
consists of a small grid of coloured squares. The typical implementation uses a 5x5 grid, 
which represents a 24 bit message, with cyan, magenta, yellow and black colours.

The tag consists of three nested components, which are from the outside moving inwards:
1. Outer quiet zone
2. Inner quiet zone
3. Tag grid

The outer quiet zone is a border region of a solid colour, usually white, and should be at least
as thick as the inner quiet zone for best detection results.  The inner quiet zone is a border 
region of a solid colour which must be:
- one on of the four colours used by the ddTag code grid, typically
black,
- the same width as the cells in the tag grid.

The grid consists of an odd numbered square grid of solid colours from a palette of 
four colours, typically cyan, magenta, yellow and black. Each cell in the grid represents two bits of data 
since it is in one of four states. ddTags officially come in the following sizes:
- 5x5
- 7x7
- 9x9
- 11x11

Odd numbers are used for the grid size since the central cell is used to encode the grid size, which 
requires an unambiguous center position. The central cell does not contain any message data. The patent uses the 
following encoding for grid sizes:

| NxN     | Center Cell |
|---------|-------------|
| 5x5     | cyan        |
| 7x7     | magenta     |
| 9x9     | yellow      |
| 11 X 11 | black       |

The corners do not contain any message data. Instead, they are used as follows:
- The bottom left cell must contain the darkest colour, e.g. black, from the colour palette as this is used to orient 
the tag.
- The other corners are used to infer the colour palette used by the tag, so they must have distinct colours.
- The other corners determine bit value associated with each colour, which starting from the top left and moving clockwise around the grid are 
00, 01, 10, 11. For example if the top left corner is cyan then all cyan cells have the value 00.

To ensure data integrity, the grid contains a CRC code, which is contained in the center row and column
excluding the center cell. The CRC length varies with the dimensions of the grid. The patent lists the following
correspondence:

| NxN     | Message length | CRC length | CRC Polynomial  |
|---------|----------------|------------|-----------------|
| 5x5     | 24 bits        | 16 bits    | CRC-16-CDMA2000 |
| 7x7     | 64 bits        | 24 bits    | CRC-24-Radix-64 |
| 9x9     | 120 bits       | 32 bits    | CRC-32Q         |
| 11 X 11 | 192 bits       | 40 bits    | CRC-40-GSM      |


## Detecting Possible Tags

The ddTag patent does not describe a particular process for detecting "frames" and leaves
it up to the implementor. However it suggests that \[2\] may be used.

In `detect_frames` we use a modified version of \[2\] as follows:
1. Convert image to grayscale
2. Detect edges by local adaptive thresholding (cv.adaptiveThreshold)
3. Detect contours by Suzuki's method (cv.findContours)
4. Fit polygon to contours (cv.approxPolyDP)
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
   1. Convert cells to binary using the rule that the palette is ordered clockwise starting at the top left with the binary values 00, 01, 10, 11. 
   2. Extract message code and CRC code. Both message and CRC are read left-to-right and top-to-bottom. 
   3. Validate message code with CRC code.

## References

\[1\]: European Patent [EP3561729NWA1][1]

\[2\]: Garrido-Jurado, S., et al. (2014). [Automatic generation and detection of highly reliable fiducial markers under occlusion][2]. Pattern Recognition.


[1]: https://data.epo.org/publication-server/rest/v1.0/publication-dates/20191030/patents/EP3561729NWA1/document.pdf
[2]: https://cs-courses.mines.edu/csci507/schedule/24/ArUco.pdf
[3]: https://www.navilens.com
[4]: https://www.ddtags.com
