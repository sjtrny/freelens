"""Serve the field-photo evaluation viewer."""

import argparse
import secrets
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
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
from PIL import Image

from scripts.evaluation_dataset import (
    DEFAULT_MANIFEST,
    LOCATION_CORNERS,
    load_dataset,
    update_tag,
)

TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[1] / "templates"
DEFAULT_PORT = 8899
RECTIFIED_TAG_SIZE = 320


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


def _location_from_query(arguments, image_size):
    location = {}
    try:
        for corner in LOCATION_CORNERS:
            location[corner] = [
                int(arguments[f"{corner}_x"]),
                int(arguments[f"{corner}_y"]),
            ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("provide eight integer location coordinates") from error

    width, height = image_size
    if any(x < 0 or x >= width or y < 0 or y >= height for x, y in location.values()):
        raise ValueError("location coordinates are outside the image")

    points = np.asarray([location[corner] for corner in LOCATION_CORNERS], dtype=float)
    edges = np.roll(points, -1, axis=0) - points
    following_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following_edges[:, 1] - edges[:, 1] * following_edges[:, 0]
    area = 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if area < 1 or not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError("location must be a non-degenerate convex region")
    return points


def _perspective_coefficients(source_points):
    edge = RECTIFIED_TAG_SIZE - 1
    destination_points = ((0, 0), (edge, 0), (edge, edge), (0, edge))
    matrix = []
    values = []
    for (output_x, output_y), (source_x, source_y) in zip(
        destination_points, source_points
    ):
        matrix.extend(
            (
                (
                    output_x,
                    output_y,
                    1,
                    0,
                    0,
                    0,
                    -source_x * output_x,
                    -source_x * output_y,
                ),
                (
                    0,
                    0,
                    0,
                    output_x,
                    output_y,
                    1,
                    -source_y * output_x,
                    -source_y * output_y,
                ),
            )
        )
        values.extend((source_x, source_y))
    return np.linalg.solve(np.asarray(matrix), np.asarray(values))


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

    @app.get("/rectified/<int:case_index>.png")
    def rectified_tag(case_index):
        with dataset_lock:
            current_dataset = dataset
        try:
            case = current_dataset.case(case_index)
        except KeyError:
            abort(404)

        try:
            source_points = _location_from_query(
                request.args,
                current_dataset.image_size(case["image"]),
            )
        except ValueError as error:
            abort(400, description=str(error))

        try:
            coefficients = _perspective_coefficients(source_points)
        except np.linalg.LinAlgError:
            abort(400, description="location cannot be rectified")
        with Image.open(current_dataset.image_path(case["image"])) as source:
            rectified = source.convert("RGB").transform(
                (RECTIFIED_TAG_SIZE, RECTIFIED_TAG_SIZE),
                Image.Transform.PERSPECTIVE,
                coefficients,
                Image.Resampling.BICUBIC,
            )
        output = BytesIO()
        rectified.save(output, "PNG")
        output.seek(0)
        response = send_file(output, mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

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
