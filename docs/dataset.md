# Dataset

The contributor-supplied photographs in `dataset/images/` use the `CC-BY-4.0` licence. Refer to the [licence file](../dataset/LICENSE) or the [Creative Commons licence](https://creativecommons.org/licenses/by/4.0/) for details.

The repository does not contain the NaviLens free-kit archive. That archive is not part of the contributor photograph licence. Permission is necessary before the project can distribute it. Refer to [Testing NaviLens code PDFs](./testing-navilens-codes.md) to check a local copy.

## Field photograph benchmark

[`dataset/evaluation.json`](../dataset/evaluation.json) records the expected 5×5 tags. Each tag has these fields:

- `message`: a six-digit hexadecimal string, or `null` if the tag is confirmed but its identity is unknown.
- `conditions`: a list that describes the tag region, not the whole image.
- `location`: optional corner positions as integer `[x, y]` pairs in source-image pixels. The order is clockwise from `top_left`.
- `description`: optional text with more information about the tag.
- `scorable`: an optional flag. The default is `true`. A value of `false` keeps the tag record but removes it from the expected detector output.

The initial message labels use FreeLens detections with valid CRCs. These labels are provisional. They do not show independently that the decoder is correct.

An image with `tags: null` has no reviewed labels. A reviewer must label it independently. An image with `tags: []` has no tags, as checked by a reviewer.

To run the benchmark, use this command from the repository root:

```bash
python scripts/benchmark_dataset.py
```

The command reports scores and execution time. It exits successfully after a completed run, regardless of the scores. It is not part of pytest or continuous integration (CI). To save detailed results for comparison, use `--output results.json`.

The benchmark includes reviewed images if all their scorable tags have known messages. It omits unreviewed images and images with a null message on a scorable tag. It also omits non-scorable tags from the expected detector output.

Use the [tag editor](./tag-editor.md) to review labels and edit tag locations.

## PyCon AU 2024 contributors

- Elliana May (mause.me)
- Cait Macleod (caitelatte)
- David Vo (auscompgeek)
- Peter Hall (urcher)
- Stephen Tierney (sjtrny)

## PyCon AU 2025 contributors

- Kesara Rathnayake (dh90909252)
- David Vo (auscompgeek)
- Liam Bluett (infamousturtle)
- Toby Lovett (tobythetober)
- Arshia (riaar)
- Stephen Tierney (sjtrny)
