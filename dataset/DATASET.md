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
machine. Compose publishes port 8899 on all host interfaces. Drag any of a located tag's
four corner handles to update its outline and coordinate fields. Use Save to persist or
Cancel to restore all unsaved changes. Tag edits are validated and atomically replace
`evaluation.json` in the `evaluation-data` Docker volume, the container's only writable
application path. The volume persists across container and image rebuilds. Export the
edited manifest back to the project with:

```bash
docker compose cp evaluation-viewer:/app/dataset/evaluation.json dataset/evaluation.json
```

The viewer has no user authentication, so expose it only on a trusted network. Stop the
service with `docker compose down`. Removing the volume with
`docker compose down --volumes` discards edits and seeds a fresh dataset from the image
on the next start.

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
