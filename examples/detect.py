from freelens import detect_tags
from PIL import Image

img = Image.open("../dataset/images/0001.jpg")

tags_list = detect_tags(img, n=5)

for tag in tags_list:
    print(tag.message)
