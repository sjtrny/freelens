from freelens import (
    detect_frames,
    decode_frames,
)

from PIL import Image

img = Image.open("dataset/positives/PXL_20241125_120348955.jpg")

frames = detect_frames(img)

codes = decode_frames(img, frames, n=5, validate_crc=False)

print(codes)