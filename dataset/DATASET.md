# Dataset

The contributor-supplied photographs under `positives/` and `negatives/` are licensed
under `CC-BY-4.0`.

Refer to the `LICENSE` file or https://creativecommons.org/licenses/by/4.0/ for details.

The placeholder `navilens-provided/` directory is outside that license grant. Its README
records the separate provenance and redistribution merge gate for the proposed NaviLens
free-kit corpus; no corpus files are currently included.

## Field Photograph Benchmark

`evaluation.json` records the expected 5x5 tags. Each tag has a six-digit hexadecimal or
`null` `message`, a `conditions` list, and an optional four-corner `location`. A null
message records a confirmed tag whose identity cannot be determined. Locations use
integer `[x, y]` pairs in original-image pixels, clockwise from `top_left`. Conditions
describe the individual tag region, not the whole image. A message inferred from another
photograph may include `message_provenance` with `type: "related_image"` and that
image's manifest path.

The initial messages were bootstrapped from CRC-valid FreeLens detections, so they are
provisional benchmark data rather than independent proof of correctness. `tags: null`
means the image needs independent labelling; `tags: []` means it has been reviewed and
contains no tags.

Run the benchmark with:

```bash
python scripts/benchmark_dataset.py
```

The command reports scores and timing but always exits successfully after a completed
run. It is intentionally not part of pytest or CI. Use `--output results.json` to save a
detailed result that can be compared between implementations. The current end-to-end
benchmark only evaluates scorable images: reviewed empty images and images for which
every present tag has a known message. It omits unreviewed images and images containing
any null message because their decoded output cannot be compared completely.

## Evaluation Viewer

Start the viewer locally with:

```bash
python -m pip install -e ".[viewer]"
python -m scripts.view_evaluation
```

Or build and start the containerized service:

```bash
docker compose up --build
```

Open http://localhost:8899 on the Docker host, or use the host's address from another
machine. Compose publishes port 8899 on all host interfaces. Located tags are outlined
in green; selecting one changes it to red and displays its four draggable corner
handles. Click a green tag to select it, or an untagged part of the image to clear the
selection. Leaving a tag with unsaved changes prompts to save, discard, or keep editing.
Use Save to persist or Cancel to restore all unsaved changes. For a tag without a
location, use Add bounding box to create an adjustable centered rectangle. Drag inside a
box to move the whole region, or hold Shift when starting a corner drag to scale the box
proportionally around its opposite corner. When zoomed in, drag elsewhere on the image
to pan. Use Add tag beside the tag-list heading to start an unsaved tag with a centered
box ready for adjustment. Leave its message blank if it cannot be determined. Scrolling
over the image zooms around the pointer without an upper zoom limit. Tag edits also
update the square, perspective-corrected preview below the details. Tag edits are
validated and atomically replace `dataset/evaluation.json` directly in the repository
through Compose's writable `./dataset:/app/dataset` bind mount. Every other application
path remains read-only.

Compose runs as UID/GID 1000 by default. On Linux, override these values if the checkout
has a different owner. If Docker runs outside a development container, also set
`VIEWER_DATASET_PATH` to the checkout path visible to the Docker daemon. These values
may be placed in the ignored `.env` file:

```bash
VIEWER_DATASET_PATH=/daemon/path/to/freelens/dataset
VIEWER_UID=1000
VIEWER_GID=1000
```

The viewer has no user authentication, so expose it only on a trusted network. Stop the
service with `docker compose down`.

## PyCon AU 2024 Contributors

- Elliana May (mause.me)
- Cait Macleod (caitelatte)
- David Vo (auscompgeek)
- Peter Hall (urcher)
- Stephen Tierney (sjtrny)

## PyCon AU 2025 Contributors

- Kesara Rathnayake (dh90909252)
- David Vo (auscompgeek)
- Liam Bluett (infamousturtle)
- Toby Lovett (tobythetober)
- Arshia (riaar)
- Stephen Tierney (sjtrny)
