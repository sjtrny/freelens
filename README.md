*This is a work in progress and sudden, dramatic changes are expected!*

[FreeLens tag generator](https://freelens.sjtrny.com/)

# FreeLens

This project provides a reference implementation of
[NaviLens](https://www.navilens.com/) and [ddTag](https://www.ddtags.com/) for
educational or personal use. Commercial use is at your own risk as NaviLens and ddTag
may attempt to enforce their IP.

## Documentation

- [NaviLens](./docs/navilens.md)
- [ddTag specification](./docs/ddtag.md)
- [ddTag detection](./docs/ddtag-detection.md)
- [CRCs in ddTags](./docs/ddtag-crc.md)
- [Testing NaviLens code PDFs](./docs/testing-navilens-codes.md)
- [Development](./docs/development.md)

## Quickstart

### Installation

```
pip install freelens
```

### Generating Tags

Freelens generates deployed-CRC tags only at the independently verified 5×5 size.

```python
from freelens import Tag

message = "101010101011000000001011"

tag = Tag.from_message(message, n=5)

tag_img = tag.to_image()

tag_img.save("tag.png")
```

### Detecting Tags

```python
from freelens import detect_tags
from PIL import Image

img = Image.open("dataset/positives/0001.jpg")

tags_list = detect_tags(
    img,
    n=5,
    validate_crc=True,
    require_valid_crc=True,
)

for tag in tags_list:
    print(tag.message)
```
