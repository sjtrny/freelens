"""Serve the field-photo evaluation viewer."""

import argparse
import secrets
from pathlib import Path
from threading import Lock

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from scripts.evaluation_dataset import (
    DEFAULT_MANIFEST,
    LOCATION_CORNERS,
    load_dataset,
    update_tag,
)

TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[1] / "templates"
DEFAULT_PORT = 8899


def _query_index(name, count, default=None):
    value = request.args.get(name)
    if value is None:
        return default

    try:
        index = int(value)
    except ValueError:
        abort(404)
    if not 0 <= index < count:
        abort(404)
    return index


def _tag_from_form(form):
    tag = {
        "message": form.get("message", "").strip().upper(),
        "conditions": [
            condition.strip()
            for condition in form.get("conditions", "").splitlines()
            if condition.strip()
        ],
    }
    values = {
        corner: [
            form.get(f"{corner}_x", "").strip(),
            form.get(f"{corner}_y", "").strip(),
        ]
        for corner in LOCATION_CORNERS
    }
    coordinates = [coordinate for point in values.values() for coordinate in point]
    if any(coordinates):
        if not all(coordinates):
            raise ValueError(
                "provide all eight location coordinates or leave all blank"
            )
        try:
            tag["location"] = {
                corner: [int(coordinate) for coordinate in point]
                for corner, point in values.items()
            }
        except ValueError as error:
            raise ValueError("location coordinates must be integers") from error
    return tag


def create_app(manifest_path=DEFAULT_MANIFEST):
    dataset = load_dataset(manifest_path)
    dataset_lock = Lock()
    app = Flask(__name__, template_folder=TEMPLATE_DIRECTORY)
    app.config["CSRF_TOKEN"] = secrets.token_urlsafe(32)

    def edit_error(message, status=400):
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify(error=message), status
        abort(status, description=message)

    @app.get("/")
    def index():
        with dataset_lock:
            current_dataset = dataset
        image_index = _query_index("image", len(current_dataset.cases), default=0)
        case = current_dataset.case(image_index)
        tags = case["tags"]
        tag_index = _query_index("tag", len(tags or []))
        selected_tag = None if tag_index is None else tags[tag_index]
        width, height = current_dataset.image_size(case["image"])

        return render_template(
            "evaluation.html",
            cases=current_dataset.cases,
            case=case,
            image_index=image_index,
            image_width=width,
            image_height=height,
            selected_tag=selected_tag,
            selected_tag_index=tag_index,
            csrf_token=app.config["CSRF_TOKEN"],
        )

    @app.get("/image/<int:case_index>")
    def image(case_index):
        with dataset_lock:
            current_dataset = dataset
        try:
            case = current_dataset.case(case_index)
        except KeyError:
            abort(404)
        return send_file(current_dataset.image_path(case["image"]), conditional=True)

    @app.post("/tag/<int:case_index>/<int:tag_index>")
    def edit_tag(case_index, tag_index):
        nonlocal dataset
        if not secrets.compare_digest(
            request.form.get("csrf_token", ""), app.config["CSRF_TOKEN"]
        ):
            return edit_error("The edit token is invalid; refresh and try again.", 403)

        try:
            tag = _tag_from_form(request.form)
            with dataset_lock:
                dataset = update_tag(dataset, case_index, tag_index, tag)
        except KeyError:
            return edit_error("Unknown image or tag.", 404)
        except ValueError as error:
            return edit_error(str(error))
        except OSError as error:
            return edit_error(f"Unable to write evaluation.json: {error}", 500)

        target = url_for("index", image=case_index, tag=tag_index)
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify(url=target)
        return redirect(target, code=303)

    return app


app = create_app()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
