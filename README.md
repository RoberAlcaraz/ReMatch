# ReMatch

**Re-identification of patterned species in open-set scenarios by matching keypoints and lines**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ReMatch identifies individual animals from photographs of their natural patterns —
lizard scale mosaics, zebra stripes, seal pelage, shark spots, human fingerprints
alike. It segments the individual, matches keypoints **and** line segments between
every pair of images with [GlueStick](https://github.com/cvg/GlueStick), and turns
those correspondences into a calibrated probability that two photographs show the
same animal.

Its distinguishing feature is **open-set operation**. Unlike approaches that assume
a fixed set of known individuals, ReMatch supports the detection of previously
unseen individuals through a learned identity-level threshold, with minimal manual
annotation — so a query can be answered *"this is a new animal"* rather than being
forced onto the closest known match.

Across five heterogeneous datasets (Balearic wall lizards, plains zebras, Saimaa
ringed seals, whale sharks and human fingerprints) ReMatch achieves an average
closed-set Top-1 accuracy of **88.23 %** (Top-5: 90.22 %) and an average open-set
accuracy of **85.78 %**. Performance is pattern-dependent: ReMatch excels on
fine-scale local patterns such as scale mosaics and fingerprints (up to 98.51 %)
and remains competitive under viewpoint variation, while other deep-embedding
models retain an advantage on datasets dominated by global cues, such as whale
sharks.

> **Paper** — Alcaraz, Amores, Villa, Marcos, Tavecchia, Igual & Rotger,
> *ReMatch: Re-identification of patterned species in open-set scenarios by
> matching keypoints and lines*, **Pattern Recognition** (Elsevier), 2026.
> Accepted, in press — DOI to follow.

---

## Contents

- [Try it first](#try-it-first)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Data layout](#data-layout)
- [Training a model](#training-a-model)
- [Identifying new images](#identifying-new-images)
- [Adapting to a new species](#adapting-to-a-new-species)
- [Configuration](#configuration)
- [Running on a cluster](#running-on-a-cluster)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Try it first

Two notebooks run the method end to end on a small dataset bundled in
[`data/`](data/): 12 Balearic wall lizards in the gallery, and a query batch of 20
photographs of which **seven show animals the model has never seen**.

| | |
|---|---|
| [`demo-1-training.ipynb`](demo-1-training.ipynb) | Match every pair, train the classifier, meta-model and threshold |
| [`demo-2-query.ipynb`](demo-2-query.ipynb) | Identify the new batch, reject the unseen animals, score the result |

```bash
pip install -r requirements.txt
jupyter lab demo-1-training.ipynb
```

They read their configuration from `params/`, their helpers from `utils/`, and
write into `results/` — the same paths the pipeline scripts use, so the demo *is*
the pipeline rather than a parallel implementation. No model weights are needed:
GlueStick and SuperPoint download themselves, and the bundled images are already
pattern crops. See [`data/README.md`](data/README.md).

The images come from **BalearicLizard**, published separately:

> Alcaraz, R., Albalat-Oliver, B., Villa, A. *et al.* A long-term photographic
> dataset for individual identification of the Balearic wall lizard.
> *Scientific Data* (2026).
> [doi:10.1038/s41597-026-07411-z](https://doi.org/10.1038/s41597-026-07411-z)
> · [Kaggle](https://www.kaggle.com/datasets/roberalcaraz/baleariclizard)

---

## How it works

Five stages, common to every dataset:

1. **ROI segmentation** — isolate the animal from its background.
2. **Wireframe extraction** — describe it with SuperPoint keypoints *and* LSD line
   segments, merged into a single wireframe per image.
3. **Pair-level feature aggregation** — match two wireframes with GlueStick, drop
   matches scoring below 0.2, and reduce what survives to four numbers:
   `num_nonzero_points`, `mean_point_prob`, `num_nonzero_lines`, `mean_line_prob`.
4. **Pair-level classification** — Logistic Regression, Random Forest, XGBoost and
   CatBoost are compared over image-disjoint rolling folds; the best mean F1 wins.
   Output: a probability `p` that a pair shows the same individual.
5. **Identity-level calibration** — for a query and a candidate identity, `p` is
   aggregated over all that identity's photographs into `[p, max, mean]` and passed
   through a logistic-regression meta-model. If the best score falls below the
   threshold **τ\***, the query is reported as a new individual.

One stage is optional. **Pattern extraction** (SAM + structured edge detection)
further isolates fine-scale texture and is worth enabling for species patterned
like lizards; the other four datasets in the paper match on the segmented ROI
directly.

Stages 4 and 5 are **retrained per dataset**. That is a design choice, not a
limitation — the models are small and the matching stack above them never
changes.

---

## Installation

```bash
git clone https://github.com/RoberAlcaraz/ReMatch.git
cd ReMatch
conda create -n rematch python=3.13
conda activate rematch
pip install -r requirements.txt
```

Two pins are not interchangeable with their obvious alternatives:
**`opencv-contrib-python`** (not `opencv-python`) because pattern extraction calls
`cv2.ximgproc`, and **`scikit-learn==1.7.2`** because trained models are stored as
scikit-learn pickles.

Grounded-SAM segmentation additionally needs GroundingDINO, which is not on PyPI:

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

A CUDA-capable GPU is strongly recommended. Everything falls back to CPU, but
matching every pair of a real database is impractically slow without one.

### Model weights

Place these in `models/`:

| File | What for | Source |
|---|---|---|
| `sam_vit_h_4b8939.pth` | SAM ViT-H — segmentation and pattern extraction | [segment-anything releases](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| `groundingdino_swint_ogc.pth` | GroundingDINO — text-prompted detection | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `model.yml.gz` | Structured edge detection — pattern extraction only | [OpenCV contrib](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz) |
| `yolo-segmentation.pt` | Optional: fine-tuned YOLO detector for lizards | [Hugging Face](https://huggingface.co/roberalcaraz/lizard-body-segmentation) |

GlueStick and SuperPoint weights download themselves on first run. Behind a
TLS-intercepting proxy that can fail; fix your trust store, drop the weights into
`resources/weights/` by hand, or as a last resort set `REMATCH_INSECURE_SSL=1`.

---

## Data layout

Training images go in one directory per individual — **the directory name is the
identity label**:

```
data/images/
├── individual_001/
│   ├── IMG_0001.jpg
│   └── IMG_0002.jpg
└── individual_002/
    └── …
```

Each batch of new photographs to identify goes in its own directory:

```
data/new/Batch1/
├── IMG_1001.jpg
└── IMG_1002.jpg
```

---

## Training a model

Run from the repository root, in order. Each step reads what the previous one
wrote.

```bash
# 1. Segment every image, and optionally extract patterns.
python scripts/P1-image_preparation.py
```
Writes `data/images-segmented/`, `results/unique_ids.txt`, and — with
`STEP_1B = True` — `data/images-pattern/`.
*Review the segmentations before continuing; a bad mask wastes the whole matching run.*

```bash
# 2. Compute wireframes and match every pair.
python scripts/P2-pattern_matching.py
```
Writes `results/precomputed_wireframe.h5` and `results/matches.lmdb/`. **This is
the expensive step** — cost grows quadratically with database size. Both caches
are resumable, so rerunning skips work already done.

```bash
# 3. Aggregate matches into the pair-level feature table.
python scripts/P3-feature_aggregation.py
```
Writes `results/processed_matches.parquet`.

```bash
# 4. Train the classifier, meta-model and threshold.
python scripts/P4-model_training.py
```
Writes `results/best_classification_model.pkl`,
`results/logistic_regression_model.pkl`, `results/scaler.pkl` and
`results/threshold.txt`. These four files are your model.

---

## Identifying new images

```bash
export NEW_IMAGES_NAME="Batch1"     # which directory under data/new/ to process

# 1. Segment the new images.
python scripts/Q1-image_preparation.py
```
*Review `data/new/Batch1_checks/` and delete failures from
`data/new/Batch1-pattern/` before continuing.*

```bash
# 2. Match against the database and rank candidates.
python scripts/Q2-pattern_matching.py
python scripts/Q3-feature_aggregation.py
python scripts/Q4-model_application.py
```
Writes `results/top10_results_Batch1.csv` — the ten best candidate identities per
query, with a calibrated probability each. Queries scoring below τ\* are marked
`new`.

*Review that file: confirm the matches and assign identities to genuinely new
animals.*

```bash
# 3. Merge the reviewed batch into the database.
python scripts/Q5-add_results_to_db.py
```
Adds the new images, wireframes and matches to the database, so the gallery grows
as you work.

The two review checkpoints are deliberate. ReMatch ranks candidates for an
expert; it does not silently commit identity decisions.

---

## Adapting to a new species

1. **Start with Grounded-SAM.** Set `SEGMENTATION_MODEL = "GroundedSAM"` and
   `CLASSES = ["your animal"]` in `params/image_preparation_params.py`. It is
   text-prompted, so it needs no per-species training — the right thing to try
   first. The paper uses this route for zebras, seals and sharks.
2. **If segmentation struggles**, fine-tune a YOLO segmentation model on a few
   hundred annotated images, point `YOLO_SEGMENTATION_MODEL` at it and set
   `SEGMENTATION_MODEL = "YOLO"`. A bounded, one-time annotation cost — this is
   the route used for lizards.
3. **Pattern extraction** (`STEP_1B`) isolates fine-scale texture with SAM and
   structured edges. Leave it on for lizard-like scale mosaics; turn it off if
   the whole segmented animal is the pattern, as with stripes or spots.
4. **Retrain.** Run the four `P` scripts on your labelled images. Only the two
   small models change — the matching stack never does.

---

## Configuration

Everything lives in two files:

- **`params/params.py`** — GlueStick and wireframe settings, `IMAGE_HEIGHT_RESIZE`
  (670 px; the models are calibrated against it, so changing it means retraining),
  and every data and result path.
- **`params/image_preparation_params.py`** — `SEGMENTATION_MODEL`
  (`"GroundedSAM"` or `"YOLO"`), `CLASSES`, detection thresholds, and the step
  toggles `STEP_1A` (segmentation) and `STEP_1B` (pattern extraction).

`NEW_IMAGES_NAME` is read from the environment and selects which batch the `Q`
scripts operate on.

---

## Running on a cluster

The scripts are plain Python and take no arguments, so a job script is three
lines. For SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=rematch-match
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

conda activate rematch
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NEW_IMAGES_NAME="${1:-Batch1}"   # only needed for the Q scripts

python scripts/P2-pattern_matching.py
python scripts/P3-feature_aggregation.py
```

Give step 2 the most time and CPUs — it dominates. Everything else is minutes.
Always run from the repository root: paths in `params/` are relative to it.

---

## Project structure

```
ReMatch/
├── scripts/          P1–P4 training, Q1–Q5 query — the entry points
├── params/           Configuration
├── utils/            Wireframes, matching, segmentation, pattern extraction
├── gluestick/        Vendored GlueStick (MIT) — see gluestick/NOTICE.md
├── demo-1-training.ipynb   Walk through training, on the bundled lizard data
├── demo-2-query.ipynb      Walk through identification, including open-set rejection
├── data/             Your images, plus the bundled lizard demo set
├── models/           Model weights — downloaded, not tracked
├── results/          Pipeline outputs and trained models — not tracked
└── licenses/         Third-party license texts
```

---

## Citation

```bibtex
@article{alcaraz2026rematch,
  title   = {{ReMatch}: Re-identification of patterned species in open-set
             scenarios by matching keypoints and lines},
  author  = {Alcaraz, Roberto and Amores, Angel and Villa, Alejandro and
             Marcos, Marta and Tavecchia, Giacomo and Igual, Jos{\'e} Manuel and
             Rotger, Andreu},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {In press}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Please also cite
GlueStick, on which the matching stage is built — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## License

MIT — see [LICENSE](LICENSE). Bundled and runtime third-party components carry
their own terms, including vendored GlueStick code (MIT), the SAM-derived mask
generator (Apache-2.0) and Ultralytics YOLO (**AGPL-3.0**, with obligations of its
own if you redistribute a service built on it). All documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
