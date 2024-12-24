from freelens import detect_tags
from PIL import Image

img = Image.open("../dataset/positives/PXL_20241124_081401367.MP.jpg")

tags_list = detect_tags(img, n=5)

for tag in tags_list:
    print(tag.message)
