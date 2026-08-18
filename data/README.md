# `data/`

Your workspace, plus the small bundled dataset the demo notebooks run on.

```
data/
├── images/<individual>/            Raw labelled photographs — directory name IS the identity
├── images-segmented/<individual>/  Segmentation output (scripts/P1)
├── images-pattern/<individual>/    Pattern extraction output (scripts/P1) — what ReMatch matches on
├── new/<batch>/                    A batch of photographs to identify
└── new/<batch>-pattern/            Their pattern crops
```

Nothing here is tracked by git **except** the bundled demo set — the `lizard_*`
directories. Add your own images under any other name and they stay out of
version control.

## The bundled demo set

| | |
|---|---|
| Gallery | 12 individuals × 4 photographs = 48, in `images-pattern/lizard_*/` |
| Query batch | 20 photographs in `new/lizard_demo-pattern/` |
| — of individuals in the gallery | 13 |
| — of individuals **never seen** | 7, across 4 animals |

That last row is the point: it lets the demo show open-set behaviour, where the
pipeline must reject a query rather than force it onto the nearest known animal.
`lizard_demo_ground_truth.json` records the true identity of every query image
and whether it is in the gallery — used only to score the result at the end.

Run it with [`demo-1-training.ipynb`](../demo-1-training.ipynb) followed by
[`demo-2-query.ipynb`](../demo-2-query.ipynb).

### These are pattern crops, not raw photographs

The bundled images are the output of `scripts/P1` — the lizard is segmented with
YOLO, then its ventral scale mosaic is isolated with SAM and structured edge
detection. That is the stage the demos skip, because it needs the original field
photographs. The crops are what ReMatch actually matches on, so the demos begin
from them.

To run the segmentation stage yourself, download the full dataset (below) into
`data/images/<individual>/` and run `python scripts/P1-image_preparation.py`.

### Source and licence

The images come from **BalearicLizard**: 4 619 photographs of 1 009 individual
Balearic wall lizards (*Podarcis lilfordi*), collected over fifteen years of
capture–recapture monitoring on Illot d'en Curt, Mallorca, and identified by
expert observers from their ventral scale patterns.

> Alcaraz, R., Albalat-Oliver, B., Villa, A. *et al.* **A long-term photographic
> dataset for individual identification of the Balearic wall lizard.**
> *Scientific Data* (2026).
> [doi:10.1038/s41597-026-07411-z](https://doi.org/10.1038/s41597-026-07411-z)

**Download the full dataset**:
[kaggle.com/datasets/roberalcaraz/baleariclizard](https://www.kaggle.com/datasets/roberalcaraz/baleariclizard)

The 68 crops here are a subset, downscaled to 900 px high. Identity labels
(`lizard_C116`, `lizard_C373`, …) are the dataset's own annotations with a
`lizard_` prefix, which is what keeps them distinguishable from your own data in
`.gitignore`.
