from freelens import (
    message_length_for_N,
    max_int_for_N,
    tag_colours,
    make_tag_image,
)
import random

# Use a standard 5x5 sized ddTag
N = 5

# Determine the message length and max representable integer
# for a tag of this size
message_len = message_length_for_N(N)
max_int = max_int_for_N(N)
print(f"For a {N}x{N} tag, the maximum message length is {message_len} and largest integer is {max_int}.")

# Select a random integer between 0 and max_int as the data
random.seed(0)
id_int = random.randint(0, max_int)
print(f"The tag will represent the integer {id_int}")

# Get a list that represents the cell colours of the tag using default palette
tag_colours = tag_colours(id_int)

# Convert the cell colour list to a PIL Image
tag_img = make_tag_image(tag_colours)

# Save to disk
tag_img.save("tag.png")

