import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.benchmark_dataset import benchmark_dataset, summarize
from scripts.evaluation_dataset import add_tag, load_dataset, update_tag
from scripts.view_evaluation import create_app


@pytest.fixture
def evaluation_manifest(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (100, 80), "white").save(tmp_path / "images/tagged.jpg")
    Image.new("RGB", (40, 30), "black").save(tmp_path / "images/empty.jpg")
    Image.new("RGB", (60, 50), "gray").save(tmp_path / "images/review.jpg")

    manifest = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "image": "images/tagged.jpg",
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
                            "scorable": False,
                        },
                    ],
                },
                {"image": "images/empty.jpg", "tags": []},
                {
                    "image": "images/review.jpg",
                    "tags": [
                        {
                            "message": None,
                            "conditions": ["severe_blur"],
                        }
                    ],
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
    assert dataset.cases[0]["tags"][1]["scorable"] is False
    assert dataset.cases[2]["tags"] == [
        {"message": None, "conditions": ["severe_blur"]}
    ]
    assert dataset.image_size("images/tagged.jpg") == (100, 80)


def test_load_dataset_preserves_an_unreviewed_case(evaluation_manifest):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    cases[2]["tags"] = None
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")

    assert load_dataset(evaluation_manifest).case(2)["tags"] is None


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
        (
            lambda cases: cases[0]["tags"][0].update(scorable="false"),
            "invalid scorable value",
        ),
    ),
)
def test_load_dataset_rejects_invalid_tags(evaluation_manifest, change, message):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    change(cases)
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(evaluation_manifest)


def test_load_dataset_rejects_image_level_conditions(evaluation_manifest):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    cases[2]["conditions"] = ["severe_blur"]
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match="conditions .* must be attached to tags"):
        load_dataset(evaluation_manifest)


def test_load_dataset_normalizes_and_validates_a_tag_description(evaluation_manifest):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    cases[0]["tags"][0]["description"] = "  Message seen in another image.  "
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")

    tag = load_dataset(evaluation_manifest).case(0)["tags"][0]
    assert tag["description"] == "Message seen in another image."

    cases[0]["tags"][0]["description"] = 123
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid description"):
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


def test_add_tag_appends_to_an_unreviewed_case(evaluation_manifest):
    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    cases[2]["tags"] = None
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")
    dataset = add_tag(
        load_dataset(evaluation_manifest),
        2,
        {
            "message": "123ABC",
            "conditions": [],
            "location": {
                "top_left": [15, 12],
                "top_right": [44, 12],
                "bottom_right": [44, 37],
                "bottom_left": [15, 37],
            },
        },
    )

    assert dataset.case(2)["tags"] == [
        {
            "message": "123ABC",
            "conditions": [],
            "location": {
                "top_left": [15, 12],
                "top_right": [44, 12],
                "bottom_right": [44, 37],
                "bottom_left": [15, 37],
            },
        }
    ]
    assert load_dataset(evaluation_manifest).case(2)["tags"] == dataset.case(2)["tags"]


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

    assert results[0]["expected"] == ["AABBCC"]
    assert results[0]["actual"] == ["AABBCC", "AABBCC"]
    assert results[0]["unexpected"] == []
    assert results[0]["status"] == "pass"
    assert results[1]["status"] == "pass"
    assert len(results) == 2
    assert all(result["image"] != "images/review.jpg" for result in results)
    summary = summarize(results)
    assert summary["scorable_images"] == 2
    assert summary["positive_images"] == 1
    assert summary["negative_images"] == 1
    assert "manual_review_images" not in summary


def test_viewer_selects_a_tag_and_draws_its_location(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b"AABBCC" in response.data
    assert b"occluded" in response.data
    assert (
        b'<polygon data-overlay-polygon data-tag-region data-tag-url="/?image=0&amp;tag=1" '
        b"data-bounding-region "
        b'points="10,10 90,10 90,70 10,70">' in response.data
    )
    assert b'class="tag-overlay is-selected"' in response.data
    assert b'data-tag-index="1" data-selected' in response.data
    assert b".tag-overlay polygon { cursor: pointer; fill: #34c75922" in response.data
    assert (
        b".tag-overlay.is-selected polygon { cursor: move; fill: #ff3b3033"
        in response.data
    )
    assert response.data.count(b'class="corner-handle"') == 4
    assert b'data-initial-location="true"' in response.data
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
        "scorable": False,
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
        b'<polygon data-overlay-polygon data-tag-region data-tag-url="/?image=0&amp;tag=1" '
        b"data-bounding-region "
        b'points="5,6 70,6 70,60 5,60">' in updated_page
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
    assert b"Leave blank when the message cannot be determined." in response.data
    assert b"Description" in response.data
    assert b'name="description"' in response.data
    assert b'name="scorable" value="true">' in response.data
    assert b"Include in scoring" in response.data
    assert b'name="conditions"' in response.data
    assert b">occluded</textarea>" in response.data
    assert b'name="top_left_x" value="10" min="0" max="99"' in response.data
    assert b'name="top_left_y" value="10" min="0" max="79"' in response.data
    assert b'<button type="submit">Save</button>' in response.data
    assert b'<button type="reset" disabled>Cancel</button>' in response.data
    assert b'event.target.closest("[data-tag-form]")' in response.data
    assert b"control.checked !== control.defaultChecked" in response.data
    assert b"body: new FormData(form)" in response.data
    assert b'method: "POST"' in response.data
    assert b'event.target.closest("[data-corner-handle]")' in response.data
    assert b"form.elements.namedItem(`${corner}_x`).value = nextX" in response.data
    assert b'overlay.querySelector("[data-overlay-polygon]")' in response.data
    assert b"hold Shift to scale the box" in response.data
    assert b"const scaleBox = event.shiftKey" in response.data
    assert b'top_left: "bottom_right"' in response.data
    assert b"const requestedScale = (" in response.data
    assert b"Math.min(maximumScale, requestedScale)" in response.data
    assert b"updateCorner(point, nextX, nextY, false)" in response.data
    assert b'document.addEventListener("reset"' in response.data
    assert b"form.requestSubmit()" not in response.data
    assert b'showStatus(form, "Changes discarded.", false, true)' in response.data
    assert (
        b"form.querySelector('button[type=\"reset\"]').disabled = !dirty"
        in response.data
    )
    assert b'status.classList.add("is-fading"), 10000' in response.data
    assert b".form-status.is-fading { opacity: 0; }" in response.data
    assert b'event.target.closest("[data-add-location]")' in response.data
    assert b"function defaultLocationInView(stage)" in response.data
    assert b"Math.max(stageBounds.left, viewportBounds.left)" in response.data
    assert b"Math.min(imageWidth, imageHeight) * 0.25 / zoom" in response.data
    assert b"const right = left + sideLength - 1" in response.data
    assert b"applyDefaultLocation(nextForm, true)" in response.data
    assert b'overlay?.dataset.initialLocation === "false"' in response.data
    assert b'event.target.closest("[data-bounding-region]")' in response.data
    assert b"imageWidth - 1 - maximumX" in response.data
    assert b'event.target.closest("[data-image-stage] img")' in response.data
    assert b"viewport.scrollLeft = startScrollLeft" in response.data
    assert (
        b'class="tag-preview" data-tag-preview data-url="/rectified/0.png"'
        in response.data
    )
    assert b">Rectified tag</h2>" in response.data
    assert b"scheduleTagPreview(form)" in response.data
    assert b"image.src = url" in response.data
    assert response.data.index(b'class="form-actions"') < response.data.index(
        b"data-form-status"
    )
    assert response.data.index(b"</form>") < response.data.index(b'class="tag-preview"')


def test_viewer_handles_missing_locations_and_review_states(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    missing_location = client.get("/?image=0&tag=0").data
    assert b'data-selected data-initial-location="false" hidden' in missing_location
    assert (
        b'<polygon data-overlay-polygon data-tag-region data-tag-url="/?image=0&amp;tag=0" '
        b"data-bounding-region></polygon>" in missing_location
    )
    assert (
        b'<div class="tag-overlay" data-tag-overlay data-tag-index="1">'
        in missing_location
    )
    assert b'points="10,10 90,10 90,70 10,70"' in missing_location
    assert missing_location.count(b'class="corner-handle"') == 4
    assert (
        b'<button class="add-location" type="button" data-add-location>'
        b"Add bounding box</button>" in missing_location
    )
    assert b'name="top_left_x" value=""' in missing_location
    assert b"No location" in missing_location
    assert b".tag-preview img[hidden] { display: none; }" in missing_location
    assert b"No tags" in client.get("/?image=1").data

    unknown_message = client.get("/?image=2&tag=0").data
    assert b"Unknown message" in unknown_message
    assert b'name="message" value=""' in unknown_message
    assert b">severe_blur</textarea>" in unknown_message

    cases = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    cases[2]["tags"] = None
    evaluation_manifest.write_text(json.dumps(cases), encoding="utf-8")
    unreviewed_app = create_app(evaluation_manifest)
    unreviewed_app.config.update(TESTING=True)
    assert b"Not reviewed" in unreviewed_app.test_client().get("/?image=2").data


def test_viewer_saves_a_tag_with_an_unknown_message(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/tag/2/0",
        data={
            "csrf_token": app.config["CSRF_TOKEN"],
            "message": "",
            "description": "Too blurred to identify.",
            "scorable": "true",
            "conditions": "severe_blur",
        },
    )

    assert response.status_code == 303
    assert load_dataset(evaluation_manifest).case(2)["tags"][0] == {
        "message": None,
        "description": "Too blurred to identify.",
        "conditions": ["severe_blur"],
    }
    assert b"Unknown message" in client.get(response.headers["Location"]).data


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
    assert b'stage.addEventListener("wheel"' in response.data
    assert b"{ passive: false }" in response.data
    assert b"stage.dataset.zoom = zoom" in response.data
    assert b"maximumZoom" not in response.data
    assert b"zoomIn.disabled" not in response.data


def test_viewer_uses_partial_navigation_to_preserve_list_scroll(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0")

    assert response.status_code == 200
    assert b'event.target.closest(".image-list a, .tag-list a")' in response.data
    assert b"await fetch(url" in response.data
    assert b"new DOMParser()" in response.data
    assert b'document.querySelector("main").replaceWith(nextMain)' in response.data
    assert b'stage.querySelectorAll("[data-tag-overlay]")' in response.data
    assert b'nextMain.querySelectorAll("[data-tag-overlay]")' in response.data
    assert b'activeImage?.scrollIntoView({ block: "nearest" })' in response.data
    assert b'history.pushState(null, "", response.url)' in response.data
    assert b'window.addEventListener("popstate"' in response.data


def test_viewer_selects_overlays_and_guards_unsaved_navigation(
    evaluation_manifest,
):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0&tag=0")

    assert response.status_code == 200
    assert b'data-deselect-url="/?image=0"' in response.data
    assert b'data-tag-region data-tag-url="/?image=0&amp;tag=1"' in response.data
    assert b"pointer-events: all" in response.data
    assert b'class="unsaved-dialog" data-unsaved-dialog' in response.data
    assert b">Save and continue</button>" in response.data
    assert b">Discard changes</button>" in response.data
    assert b">Keep editing</button>" in response.data
    assert b'event.target.closest("[data-tag-region]")' in response.data
    assert b"image.dataset.deselectUrl" in response.data
    assert b"function formIsDirty(form)" in response.data
    assert b"function requestNavigation(url)" in response.data
    assert b"unsavedDialog.showModal()" in response.data
    assert b'if (action === "discard") navigate(destination)' in response.data
    assert b"saveTag(form, destination)" in response.data


def test_viewer_opens_a_new_tag_draft_with_a_default_location(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()
    original = evaluation_manifest.read_bytes()

    response = client.get("/?image=0&tag=new")

    assert response.status_code == 200
    assert b"<h2>Tags</h2>" in response.data
    assert (
        b'<button class="add-tag" type="button" '
        b'data-new-tag-url="/?image=0&amp;tag=new">Add tag</button>' in response.data
    )
    assert response.data.index(b"<h2>Tags</h2>") < response.data.index(
        b">Add tag</button>"
    )
    assert b"<span>New tag</span>" in response.data
    assert (
        b'action="/tag/0" method="post" data-tag-form data-new-tag-form'
        in response.data
    )
    assert b'name="message" value=""' in response.data
    assert b'name="scorable" value="true" checked>' in response.data
    assert (
        b'data-tag-index="2" data-selected data-initial-location="true"'
        in response.data
    )
    assert b'points="40,30 59,30 59,49 40,49"' in response.data
    assert response.data.count(b'class="corner-handle"') == 4
    assert b'name="top_left_x" value="40"' in response.data
    assert b'name="bottom_right_y" value="49"' in response.data
    assert b'event.target.closest("[data-new-tag-url]")' in response.data
    assert b"newTag?.dataset.newTagUrl" in response.data
    assert b'form.hasAttribute("data-new-tag-form")' in response.data
    assert b'history.replaceState(null, "", result.url)' in response.data
    assert evaluation_manifest.read_bytes() == original


def test_viewer_saves_a_new_tag_to_the_manifest(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/tag/0",
        data={
            "csrf_token": app.config["CSRF_TOKEN"],
            "message": "123abc",
            "scorable": "true",
            "conditions": "blur",
            "top_left_x": "25",
            "top_left_y": "20",
            "top_right_x": "74",
            "top_right_y": "20",
            "bottom_right_x": "74",
            "bottom_right_y": "59",
            "bottom_left_x": "25",
            "bottom_left_y": "59",
        },
    )

    assert response.status_code == 303
    assert response.headers["Location"] == "/?image=0&tag=2"
    assert load_dataset(evaluation_manifest).case(0)["tags"][2] == {
        "message": "123ABC",
        "conditions": ["blur"],
        "location": {
            "top_left": [25, 20],
            "top_right": [74, 20],
            "bottom_right": [74, 59],
            "bottom_left": [25, 59],
        },
    }


def test_viewer_lists_tags_above_details_and_includes_resizable_panels(
    evaluation_manifest,
):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b"<title>tagged.jpg \xc2\xb7 FreeLens evaluation</title>" in response.data
    assert b"<h1>tagged.jpg</h1>" in response.data
    assert b"<details" not in response.data
    assert b'class="image-list"' in response.data
    assert (
        b'title="images/tagged.jpg" aria-current="page">tagged.jpg</a>' in response.data
    )
    assert b">images/tagged.jpg</a>" not in response.data
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
    assert b"height: 1rem" in response.data
    assert b"width: 1rem" in response.data
    assert b'.corner-handle::before { content: ""; inset: -.5rem' in response.data
    assert b"transform: translate(-115%, -115%)" in response.data
    assert b"localStorage.setItem" in response.data
    assert b'.image-list a:not([aria-current="page"]):hover' in response.data


def test_viewer_uses_ctrl_or_command_s_to_save(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    response = app.test_client().get("/?image=0&tag=1")

    assert response.status_code == 200
    assert b'event.key.toLowerCase() !== "s"' in response.data
    assert b"(!event.ctrlKey && !event.metaKey)" in response.data
    assert b"event.repeat || saving" in response.data
    assert (
        b'document.querySelector("[data-tag-form]")?.requestSubmit()' in response.data
    )
    assert (
        b"unsavedDialog.querySelector('[data-unsaved-action=\"save\"]')"
        in response.data
    )


def test_viewer_serves_only_manifest_image_indexes(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.get("/image/0")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert (
        response.data == (evaluation_manifest.parent / "images/tagged.jpg").read_bytes()
    )

    assert client.get("/image/99").status_code == 404
    assert client.get("/?image=99").status_code == 404
    assert client.get("/?image=0&tag=99").status_code == 404


def test_viewer_rectifies_current_coordinates_to_a_square(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()
    query = {
        "top_left_x": 10,
        "top_left_y": 10,
        "top_right_x": 90,
        "top_right_y": 10,
        "bottom_right_x": 90,
        "bottom_right_y": 70,
        "bottom_left_x": 10,
        "bottom_left_y": 70,
    }

    response = client.get("/rectified/0.png", query_string=query)

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.headers["Cache-Control"] == "no-store"
    with Image.open(BytesIO(response.data)) as image:
        assert image.size == (320, 320)
        assert image.getpixel((160, 160)) == (255, 255, 255)


def test_viewer_rejects_invalid_rectification_coordinates(evaluation_manifest):
    app = create_app(evaluation_manifest)
    app.config.update(TESTING=True)
    client = app.test_client()
    degenerate = {
        "top_left_x": 10,
        "top_left_y": 10,
        "top_right_x": 20,
        "top_right_y": 20,
        "bottom_right_x": 30,
        "bottom_right_y": 30,
        "bottom_left_x": 40,
        "bottom_left_y": 40,
    }
    outside = dict(degenerate, top_right_x=100)

    assert client.get("/rectified/0.png").status_code == 400
    assert client.get("/rectified/0.png", query_string=degenerate).status_code == 400
    assert client.get("/rectified/0.png", query_string=outside).status_code == 400
    assert client.get("/rectified/99.png", query_string=degenerate).status_code == 404


def test_compose_bind_mounts_the_repository_dataset_for_edits():
    compose = (Path(__file__).parents[1] / "compose.yaml").read_text(encoding="utf-8")

    assert "${VIEWER_DATASET_PATH:-./dataset}:/app/dataset" in compose
    assert "${VIEWER_UID:-1000}:${VIEWER_GID:-1000}" in compose
    assert "evaluation-data" not in compose
