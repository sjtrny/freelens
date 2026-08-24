from PIL import Image

import freelens


def test_detect_frames_uses_configured_minimum_area(monkeypatch):
    captured = {}

    def capture_area_threshold(polygons, area_threshold):
        captured["area_threshold"] = area_threshold
        return polygons

    monkeypatch.setattr(freelens, "reduce_poly_vertices", lambda contours: [])
    monkeypatch.setattr(
        freelens,
        "frame_filter_polygons_area",
        capture_area_threshold,
    )

    freelens.detect_frames(Image.new("RGB", (101, 101), "white"))

    assert captured["area_threshold"] == 1500
    assert captured["area_threshold"] == freelens.MIN_FRAME_AREA
