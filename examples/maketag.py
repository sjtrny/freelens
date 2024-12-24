from freelens import Tag

message = "101010101011000000001011"

tag = Tag.from_message(message, 5)

tag_img = tag.to_image()

tag_img.save("tag.png")
