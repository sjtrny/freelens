from freelens import (
    detect_tags,
    decode_tag_image,
)

from PIL import Image

img = Image.open("test-images/PXL_20241123_063356874.jpg")

detected_tags = detect_tags(img)
for i, tag in enumerate(detected_tags):
    decoded_id = decode_tag_image(tag)
    print(decoded_id)
    tag.save(f"detected-tag-{decoded_id}-{i}.png")
