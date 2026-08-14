
## Detecting Possible Tags

The ddTag patent does not describe a particular process for detecting "frames" and leaves
it up to the implementor. However it suggests that \[1\] may be used.

In `detect_frames` we use a modified version of \[1\] as follows:
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
6. Validate the deployed CRC when processing a 5×5 tag and validation is enabled
   1. Convert cells to binary using the rule that the palette is ordered clockwise starting at the top left with the binary values `00`, `01`, `10`, `11`.
   2. Extract message code and CRC code.
   3. Validate message code with CRC code.


## References

\[1\]: Garrido-Jurado, S., et al. (2014). [Automatic generation and detection of highly reliable fiducial markers under occlusion][1]. Pattern Recognition.

[1]: https://cs-courses.mines.edu/csci507/schedule/24/ArUco.pdf