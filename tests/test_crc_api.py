import numpy as np
import pytest
from PIL import Image

import freelens


def _image_and_polygon():
    image = Image.new("RGB", (224, 224), "black")
    polygon = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
    return image, polygon


def test_decode_frames_passes_validate_crc_to_tag(monkeypatch):
    image, polygon = _image_and_polygon()
    calls = []

    class FakeTag:
        def __init__(self, bit_string, n, *, validate_crc):
            calls.append((bit_string, n, validate_crc))
            self.crc_valid = None

    monkeypatch.setattr(freelens, "Tag", FakeTag)
    tags = freelens.decode_frames(image, [polygon], n=5, validate_crc=False)

    assert len(tags) == 1
    assert calls == [("00" * 25, 5, False)]


def test_decode_frames_can_require_a_valid_crc_and_corners(monkeypatch):
    image, polygon = _image_and_polygon()
    results = iter(((False, True), (True, False), (True, True)))

    class FakeTag:
        def __init__(self, bit_string, n, *, validate_crc):
            self.crc_valid, self.corners_valid = next(results)

    monkeypatch.setattr(freelens, "Tag", FakeTag)
    tags = freelens.decode_frames(
        image,
        [polygon, polygon, polygon],
        validate_crc=True,
        require_valid_crc=True,
    )

    assert len(tags) == 1
    assert tags[0].crc_valid is True
    assert tags[0].corners_valid is True


def test_decode_frames_strict_validation_rejects_a_uniform_frame():
    image, polygon = _image_and_polygon()

    tags = freelens.decode_frames(
        image,
        [polygon],
        validate_crc=True,
        require_valid_crc=True,
    )

    assert tags == []


@pytest.mark.parametrize(
    ("n", "validate_crc", "require_valid_crc"),
    ((7, False, False), (5, True, True)),
)
def test_detect_tags_passes_crc_options_to_decode_frames(
    monkeypatch, n, validate_crc, require_valid_crc
):
    image = object()
    polygons = [object()]
    captured = {}

    monkeypatch.setattr(freelens, "detect_frames", lambda actual: polygons)

    def fake_decode(
        actual_image,
        actual_polygons,
        n,
        *,
        validate_crc,
        require_valid_crc,
    ):
        captured.update(
            image=actual_image,
            polygons=actual_polygons,
            n=n,
            validate_crc=validate_crc,
            require_valid_crc=require_valid_crc,
        )
        return ["tag"]

    monkeypatch.setattr(freelens, "decode_frames", fake_decode)

    result = freelens.detect_tags(
        image,
        n=n,
        validate_crc=validate_crc,
        require_valid_crc=require_valid_crc,
    )

    assert result == ["tag"]
    assert captured == {
        "image": image,
        "polygons": polygons,
        "n": n,
        "validate_crc": validate_crc,
        "require_valid_crc": require_valid_crc,
    }


@pytest.mark.parametrize("function", (freelens.decode_frames, freelens.detect_tags))
def test_require_valid_crc_requires_validation_before_image_processing(
    function, monkeypatch
):
    monkeypatch.setattr(
        freelens,
        "detect_frames",
        lambda image: pytest.fail("image processing should not run"),
    )

    with pytest.raises(
        ValueError, match="require_valid_crc=True requires validate_crc=True"
    ):
        if function is freelens.decode_frames:
            function(
                None,
                [],
                validate_crc=False,
                require_valid_crc=True,
            )
        else:
            function(
                None,
                validate_crc=False,
                require_valid_crc=True,
            )


@pytest.mark.parametrize("function", (freelens.decode_frames, freelens.detect_tags))
def test_larger_crc_validation_fails_before_image_processing(function, monkeypatch):
    monkeypatch.setattr(
        freelens,
        "detect_frames",
        lambda image: pytest.fail("image processing should not run"),
    )

    with pytest.raises(
        ValueError, match="CRC validation is supported only for 5x5 tags"
    ):
        if function is freelens.decode_frames:
            function(None, [], n=7, validate_crc=True)
        else:
            function(None, n=7, validate_crc=True)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("validate_crc", None, "validate_crc must be a bool"),
        ("require_valid_crc", None, "require_valid_crc must be a bool"),
    ),
)
def test_decode_frames_requires_boolean_options(argument, value, message):
    kwargs = {argument: value}

    with pytest.raises(TypeError, match=message):
        freelens.decode_frames(None, [], **kwargs)
