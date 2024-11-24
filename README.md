*This is a work in progress and sudden, dramatic changes are expected!*

# FreeLens

This project provides a reference implementation of [NaviLens](https://www.navilens.com) and [ddTag](https://www.ddtags.com) 
for educational or self-use purposes. Commercial use is at your own risk as NaviLens and ddTag may attempt to enforce 
their patents.

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

## ddTag

NaviLens uses a tag format called "ddTag", short for "distant dense tag". The "ddTag" brand 
is managed by a separate entity to NaviLens, however both companies are closely linked. A ddTag 
consists of a small grid of coloured squares. The typical implementation uses a 5x5 grid, 
which represents a 24 bit message, with cyan, magenta, yellow and black colours. 

### ddTag Specification

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


### Detection Pipeline

The first step to reading a ddTag is to detect tags in an image, which is achieved 
with the following steps:
1. Binary threshold the image using Otsu's method
2. Detect candidate contours in threshold image
3. Filter candidates so that they are large enough and roughly square
4. Filter candidates so that they are the innermost contours
5. Filter candidates based on conforming to spec
   1. Warp each candidate to be square (rectify)
   2. Grayscale and normalize
   3. Check that corner colours are distinct
   4. Check that bottom left corner colour matches border colour
   5. Check that each cell colour matches corner colours

This process is my own method, which is based on how QR codes are often detected.
The ddTag patent does not describe a particular process for detecting the tag and leaves
it up to the implementor.

### Decoding Pipeline

This process is adapted from the patent and assumes the tag is already rectified:
1. Inspect the center colour to determine grid size
2. Divide the image into grid cells
3. Obtain the palette colours from the four corners of the 
4. Orient the image so that the darkest corner is at the bottom left
5. Assign each cell in the grid to the closest colour in the palette (use CIE Lab colour space)
6. Convert cells to binary using the rule that the palette is ordered clockwise starting at the top left with the 
binary values 00, 01, 10, 11.
7. Extract message code and CRC code. Both message and CRC are read left-to-right and top-to-bottom.
8. Validate message code with CRC code.

TODO:
- Normalisation
- Auto detect grid size

Patent discusses using size of centre cell to determine number of cells in the grid.

The patent suggests using the black and white quiet zone of the tag to adjust the colour
of the image to correct for lighting.
- https://github.com/colour-science/colour#colour-correction-colour-characterisation


## References

- EP3561729NWA1 https://data.epo.org/publication-server/rest/v1.0/publication-dates/20191030/patents/EP3561729NWA1/document.pdf


