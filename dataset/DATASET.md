# Dataset

The contributor-supplied photographs under `positives/` and `negatives/` are licensed
under `CC-BY-4.0`.

Refer to the `LICENSE` file or https://creativecommons.org/licenses/by/4.0/ for details.

The placeholder `navilens-provided/` directory is outside that license grant. Its README
records the separate provenance and redistribution merge gate for the proposed NaviLens
free-kit corpus; no corpus files are currently included.

## Field Photograph Benchmark

`evaluation.json` records the expected 5x5 tags. Each tag has a six-digit hexadecimal
`message`, a `conditions` list, and an optional four-corner `location`. Locations use
integer `[x, y]` pairs in original-image pixels, clockwise from `top_left`. The initial
messages were bootstrapped from CRC-valid FreeLens detections, so they are provisional
benchmark data rather than independent proof of correctness. `tags: null` means the
image needs independent labelling and is excluded from message accuracy; `tags: []`
means it has been reviewed and contains no tags.

An optional case-level `conditions` list records image-wide properties such as `blur`.
Conditioned positive images are also reported as a separate challenging subset.

Run the benchmark with:

```bash
python scripts/benchmark_dataset.py
```

The command reports scores and timing but always exits successfully after a completed
run. It is intentionally not part of pytest or CI. Use `--output results.json` to save a
detailed result that can be compared between implementations. All photographs contribute
to the positive/negative detection rates; only entries with known messages contribute to
message accuracy.

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
handles. Use Save to persist or Cancel to restore all unsaved changes. For a tag without
a location, use Add bounding box to create an adjustable centered rectangle. Drag inside
a box to move the whole region; when zoomed in, drag elsewhere on the image to pan.
Scrolling over the image zooms around the pointer without an upper zoom limit. Tag edits
also update the square, perspective-corrected preview below the details. Tag edits are
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
