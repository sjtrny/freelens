import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.benchmark_dataset import benchmark_dataset
from scripts.evaluation_dataset import load_dataset, update_tag
from scripts.view_evaluation import create_app


@pytest.fixture
def evaluation_manifest(tmp_path):
    (tmp_path / "positives").mkdir()
    (tmp_path / "negatives").mkdir()
    Image.new("RGB", (100, 80), "white").save(tmp_path / "positives/tagged.jpg")
    Image.new("RGB", (40, 30), "black").save(tmp_path / "negatives/empty.jpg")
    Image.new("RGB", (60, 50), "gray").save(tmp_path / "positives/review.jpg")

    manifest = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "image": "positives/tagged.jpg",
                    "tags": [
                        {"message": "AABBCC", "conditions": []},
                        {
                            "message": "AABBCC",
                            "location": {
                                "top_left": [10, 10],
                                "top_right": [90, 10],
                                "bottom_right": [90, 70],
                                "bottom_left": [10, 70],
                            },
                            "conditions": ["occluded"],
                        },
                    ],
                },
                {"image": "negatives/empty.jpg", "tags": []},
                {
                    "image": "positives/review.jpg",
                    "tags": None,
                    "conditions": ["blur"],
                },
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def test_load_dataset_preserves_tag_identity_and_review_states(evaluation_manifest):
    dataset = load_dataset(evaluation_manifest)

    assert len(dataset.cases) == 3
    assert [tag["message"] for tag in dataset.cases[0]["tags"]] == [
        "AABBCC",
        "AABBCC",
    ]
    assert dataset.cases[1]["tags"] == []
    assert dataset.cases[2]["tags"] is None
    assert dataset.cases[2]["conditions"] == ["blur"]
    assert dataset.image_size("positives/tagged.jpg") == (100, 80)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (
            lambda cases: cases[0]["tags"][1]["location"].pop("bottom_left"),
            "invalid location",
        ),
        (
            lambda cases: cases[0]["tags"][1]["location"].update(top_left=[True, 10]),
            "invalid top_left",
        ),
        (
            lambda cases: cases[0]["tags"][1]["location"].update(
                bottom_right=[100, 70]
            ),
            "outside the image",
        ),
        (
            lambda cases: cases[0]["tags"][0].update(message="aabbcc"),
            "invalid message",
        ),
    ),
)
def test_load_dataset_rejects_invalid_tags(evaluation_manifest, change, message):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    change(cases)
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(evaluation_manifest)


def test_update_tag_validates_and_atomically_replaces_manifest(evaluation_manifest):
    original_inode = evaluation_manifest.stat().st_ino
    dataset = update_tag(
        load_dataset(evaluation_manifest),
        0,
        1,
        {
            "message": "123ABC",
            "conditions": ["blur", "occluded"],
            "location": {
                "top_left": [1, 2],
                "top_right": [80, 2],
                "bottom_right": [80, 60],
                "bottom_left": [1, 60],
            },
        },
    )

    assert evaluation_manifest.stat().st_ino != original_inode
    assert dataset.case(0)["tags"][1]["message"] == "123ABC"
    assert (
        load_dataset(evaluation_manifest).case(0)["tags"][1]
        == dataset.case(0)["tags"][1]
    )
    assert not list(evaluation_manifest.parent.glob(".evaluation.json.*.tmp"))


def test_update_tag_does_not_replace_manifest_when_validation_fails(
    evaluation_manifest,
):
    dataset = load_dataset(evaluation_manifest)
    original = evaluation_manifest.read_bytes()

    with pytest.raises(ValueError, match="outside the image"):
        update_tag(
            dataset,
            0,
            1,
            {
                "message": "123ABC",
                "conditions": [],
                "location": {
                    "top_left": [1, 2],
                    "top_right": [100, 2],
                    "bottom_right": [80, 60],
                    "bottom_left": [1, 60],
                },
            },
        )

    assert evaluation_manifest.read_bytes() == original
    assert not list(evaluation_manifest.parent.glob(".evaluation.json.*.tmp"))


def test_benchmark_reads_expected_messages_from_tags(evaluation_manifest):
    def detector(image, **options):
        assert options == {
            "n": 5,
            "validate_crc": True,
            "require_valid_crc": True,
        }
        if image.size != (100, 80):
            return []
        message = f"{int('AABBCC', 16):024b}"
        return [SimpleNamespace(message=message), SimpleNamespace(message=message)]

    results = benchmark_dataset(evaluation_manifest, detector=detector)

    assert results[0]["expected"] == ["AABBCC", "AABBCC"]
    assert results[0]["status"] == "pass"
    assert results[1]["status"] == "pass"
    assert results[2]["status"] == "manual-review"


def test_viewer_selects_a_tag_and_draws_its_location(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b"AABBCC" in response.data
    assert b"occluded" in response.data
    assert (
        b'<polygon data-overlay-polygon points="10,10 90,10 90,70 10,70">'
        in response.data
    )
    assert response.data.count(b'class="corner-handle"') == 4
    assert b'data-corner="top_left"' in response.data
    assert b'data-x="10"' in response.data
    assert b'data-y="10"' in response.data


def test_viewer_updates_a_tag_and_redirects_to_its_details(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/tag/0/1",
        data={
            "csrf_token": app.config["CSRF_TOKEN"],
            "message": "123abc",
            "conditions": "blur\n occluded \n",
            "top_left_x": "5",
            "top_left_y": "6",
            "top_right_x": "70",
            "top_right_y": "6",
            "bottom_right_x": "70",
            "bottom_right_y": "60",
            "bottom_left_x": "5",
            "bottom_left_y": "60",
        },
    )

    assert response.status_code == 303
    assert response.headers["Location"] == "/?image=0&tag=1"
    tag = json.loads(evaluation_manifest.read_text(encoding="utf-8"))[0]["tags"][1]
    assert tag == {
        "message": "123ABC",
        "conditions": ["blur", "occluded"],
        "location": {
            "top_left": [5, 6],
            "top_right": [70, 6],
            "bottom_right": [70, 60],
            "bottom_left": [5, 60],
        },
    }
    updated_page = client.get(response.headers["Location"]).data
    assert b"123ABC" in updated_page
    assert (
        b'<polygon data-overlay-polygon points="5,6 70,6 70,60 5,60">' in updated_page
    )


def test_viewer_rejects_invalid_partial_edits_without_changing_manifest(
    evaluation_manifest,
):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()
    original = evaluation_manifest.read_bytes()

    response = client.post(
        "/tag/0/1",
        data={
            "csrf_token": app.config["CSRF_TOKEN"],
            "message": "AABBCC",
            "conditions": "",
            "top_left_x": "10",
        },
        headers={"X-Requested-With": "fetch"},
    )

    assert response.status_code == 400
    assert "all eight location coordinates" in response.json["error"]
    assert evaluation_manifest.read_bytes() == original


def test_viewer_rejects_edits_without_its_csrf_token(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().post(
        "/tag/0/1",
        data={"message": "123ABC", "conditions": ""},
        headers={"X-Requested-With": "fetch"},
    )

    assert response.status_code == 403
    assert "edit token is invalid" in response.json["error"]


def test_viewer_renders_and_submits_an_editable_tag_form(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b'action="/tag/0/1" method="post" data-tag-form' in response.data
    assert b'name="csrf_token"' in response.data
    assert b'name="message" value="AABBCC"' in response.data
    assert b'name="conditions"' in response.data
    assert b">occluded</textarea>" in response.data
    assert b'name="top_left_x" value="10" min="0" max="99"' in response.data
    assert b'name="top_left_y" value="10" min="0" max="79"' in response.data
    assert b'<button type="submit">Save</button>' in response.data
    assert b'<button type="reset" disabled>Cancel</button>' in response.data
    assert b'event.target.closest("[data-tag-form]")' in response.data
    assert b"body: new FormData(form)" in response.data
    assert b'method: "POST"' in response.data
    assert b'event.target.closest("[data-corner-handle]")' in response.data
    assert b"form.elements.namedItem(`${corner}_x`).value = nextX" in response.data
    assert b'overlay.querySelector("[data-overlay-polygon]")' in response.data
    assert b'document.addEventListener("reset"' in response.data
    assert b"form.requestSubmit()" not in response.data
    assert b'showStatus(form, "Changes discarded.", false, true)' in response.data
    assert (
        b"form.querySelector('button[type=\"reset\"]').disabled = !dirty"
        in response.data
    )
    assert b'status.classList.add("is-fading"), 10000' in response.data
    assert b".form-status.is-fading { opacity: 0; }" in response.data
    assert response.data.index(b'class="form-actions"') < response.data.index(
        b"data-form-status"
    )


def test_viewer_handles_missing_locations_and_review_states(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    assert b"<polygon" not in client.get("/?image=0&tag=0").data
    assert b"No tags" in client.get("/?image=1").data
    assert b"Not reviewed" in client.get("/?image=2").data


def test_viewer_includes_fit_and_zoom_controls(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0")

    assert response.status_code == 200
    assert b'data-action="zoom-out"' in response.data
    assert b'data-action="fit"' in response.data
    assert b'data-action="zoom-in"' in response.data
    assert b'data-width="100" data-height="80"' in response.data
    assert b"imageObserver = new ResizeObserver(render)" in response.data


def test_viewer_uses_partial_navigation_to_preserve_list_scroll(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0")

    assert response.status_code == 200
    assert b'event.target.closest(".image-list a, .tag-list a")' in response.data
    assert b"await fetch(url" in response.data
    assert b"new DOMParser()" in response.data
    assert b'document.querySelector("main").replaceWith(nextMain)' in response.data
    assert b'stage.querySelector("[data-tag-overlay]")?.remove()' in response.data
    assert b'activeImage?.scrollIntoView({ block: "nearest" })' in response.data
    assert b'history.pushState(null, "", response.url)' in response.data
    assert b'window.addEventListener("popstate"' in response.data


def test_viewer_lists_tags_above_details_and_includes_resizable_panels(
    evaluation_manifest,
):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b"<details" not in response.data
    assert b'class="image-list"' in response.data
    assert b'<ul class="tag-list"' in response.data
    assert (
        response.data.index(b'id="image-list-panel"')
        < response.data.index(b"<main>")
        < response.data.index(b'id="tag-panel"')
    )
    assert response.data.index(b'id="tag-panel"') < response.data.index(
        b'class="tag-list"'
    )
    assert b'href="/?image=0&amp;tag=1"' in response.data
    assert response.data.index(b'class="tag-list"') < response.data.index(
        b">Tag details</h2>"
    )
    assert b'data-resizer="left"' in response.data
    assert b'data-resizer="right"' in response.data
    assert b"text-overflow: ellipsis" in response.data
    assert b".image-list > *, .tag-list > * { flex: none; }" in response.data
    assert b".tag-list li + li" in response.data
    assert b".tag-list a::after" not in response.data
    assert b"cursor: pointer" in response.data
    assert b"justify-content: flex-start" in response.data
    assert b"transform: translate(-115%, -115%)" in response.data
    assert b"localStorage.setItem" in response.data
    assert b'.image-list a:not([aria-current="page"]):hover' in response.data


def test_viewer_serves_only_manifest_image_indexes(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.get("/image/0")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert (
        response.data
        == (evaluation_manifest.parent / "positives/tagged.jpg").read_bytes()
    )

    assert client.get("/image/99").status_code == 404
    assert client.get("/?image=99").status_code == 404
    assert client.get("/?image=0&tag=99").status_code == 404
