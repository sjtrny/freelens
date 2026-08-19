# Dataset

The contributor-supplied photographs under `positives/` and `negatives/` are licensed
under `CC-BY-4.0`.

Refer to the `LICENSE` file or https://creativecommons.org/licenses/by/4.0/ for details.

The placeholder `navilens-provided/` directory is outside that license grant. Its README
records the separate provenance and redistribution merge gate for the proposed NaviLens
free-kit corpus; no corpus files are currently included.

## Field Photograph Benchmark

`evaluation.json` records the expected 5x5 messages as six-digit hexadecimal strings.
The initial values were bootstrapped from CRC-valid FreeLens detections, so they are
provisional benchmark data rather than independent proof of correctness. Entries with
`messages: null` need independent labelling and are excluded from message accuracy.

An optional `conditions` list records observable image properties such as `blur`.
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
