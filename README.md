# ReMatch

**A general-purpose visual re-identification pipeline for individual recognition in wildlife imagery.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ReMatch combines state-of-the-art computer vision models to segment individuals from images, extract distinctive visual patterns, and match them across observations using learned features. It supports both **training a re-identification model** from a labeled database and **querying new images** against the existing database to find matches.

> **Paper**: [ReMatch: Re-identification of patterned species in open-set scenarios by matching keypoints and lines](https://www.researchsquare.com/article/rs-7302183/v1)

---

## Pipeline overview

ReMatch operates in two modes:

### Training pipeline

Build a re-identification model from a labeled image database.

```
Raw labeled images
  │
  ├─ 1. Image preparation
  │     YOLO segmentation → SAM pattern extraction
  │     Outputs: segmented images, pattern images, unique IDs
  │
  ├─ 2. Pattern matching & feature aggregation
  │     SuperPoint wireframe descriptors → GlueStick pairwise matching → feature aggregation
  │     Outputs: wireframes (.h5), matches (.lmdb), features (.parquet)
  │
  └─ 3. Model training
        ML models + Logistic Regression meta-model + threshold optimization
        Outputs: trained models (.pkl), optimal threshold
```

### Query pipeline

Identify new individuals by matching against the existing database.

```
New unlabeled images
  │
  ├─ 1. Image preparation
  │     Same segmentation + pattern extraction as training
  │     → Manual review checkpoint
  │
  ├─ 2. Matching & prediction
  │     Wireframe computation → match against DB → feature aggregation → model prediction
  │     Outputs: top-10 candidate matches per image (.csv)
  │     → Expert review checkpoint
  │
  └─ 3. Add to database
        Merge reviewed results into the main database
        Outputs: updated DB (wireframes, matches, unique IDs)
```

---

## Project structure

```
ReMatch/
├── scripts/                  # Main pipeline scripts
│   ├── P1-image_preparation.py       # Training: segment + extract patterns
│   ├── P2-pattern_matching.py        # Training: compute wireframes + match
│   ├── P3-feature_aggregation.py     # Training: aggregate match features
│   ├── P4-model_training.py          # Training: train classification models
│   ├── Q1-image_preparation.py       # Query: segment + extract patterns
│   ├── Q2-pattern_matching.py        # Query: match against DB
│   ├── Q3-feature_aggregation.py     # Query: aggregate features
│   ├── Q4-model_application.py       # Query: apply models, rank candidates
│   └── Q5-add_results_to_db.py       # Query: merge reviewed results into DB
├── gluestick/                # GlueStick point-and-line matching module
│   └── models/               # SuperPoint, wireframe, GlueStick model definitions
├── utils/                    # Shared utilities
│   ├── utils.py              # Wireframe computation, pattern matching, feature processing
│   ├── image_preparation_utils.py    # YOLO segmentation, SAM pattern extraction
│   └── automatic_mask_and_probability_generator.py  # SAM mask generator with Sobel filtering
├── params/                   # Configuration files
│   ├── params.py             # Model configs, paths, GlueStick/wireframe settings
│   └── image_preparation_params.py   # Segmentation and pattern extraction settings
├── models/                   # Pre-trained model weights (not tracked in git)
├── data/                     # Image data (not tracked in git)
│   ├── images/               # Labeled training images: images/<individual_id>/
│   └── new/                  # New query image batches: new/<batch_name>/
├── results/                  # Pipeline outputs (not tracked in git)
├── P-step*.sh                # SLURM wrappers for training pipeline
└── Q-step*.sh                # SLURM wrappers for query pipeline
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RoberAlcaraz/ReMatch.git
cd ReMatch
```

### 2. Create environment and install dependencies

```bash
conda create -n rematch python=3.13
conda activate rematch
pip install -r requirements.txt
```

### 3. Download model weights

Place the following files in the `models/` directory:

| File | Description | Source |
|------|-------------|--------|
| `sam_vit_h_4b8939.pth` | SAM ViT-H checkpoint | [segment-anything releases](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| `yolo-segmentation.pt` | Custom fine-tuned YOLO segmentation model | User-provided |
| `model.yml.gz` | Structured edge detection model | [OpenCV contrib](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz) |

> **Note**: GlueStick and SuperPoint weights are downloaded automatically on first run.

---

## Data format

### Training data

Organize labeled images by individual identity:

```
data/images/
├── individual_001/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── individual_002/
│   └── ...
└── ...
```

### Query data

Place new images in a named batch folder:

```
data/new/<batch_name>/
├── IMG_1001.jpg
├── IMG_1002.jpg
└── ...
```

---

## Usage

### Training pipeline

Run the three training steps sequentially:

```bash
# Step 1: Segment images and extract patterns
python scripts/P1-image_preparation.py

# Step 2: Compute wireframes, match all pairs, and aggregate features
python scripts/P2-pattern_matching.py
python scripts/P3-feature_aggregation.py

# Step 3: Train classification models
python scripts/P4-model_training.py
```

**Outputs** (in `results/`):
- `best_classification_model.pkl` — trained Random Forest model
- `logistic_regression_model.pkl` — meta-model for probability calibration
- `scaler.pkl` — feature scaler
- `threshold.txt` — optimal classification threshold

For detailed information, see [README_training_pipeline.md](README_training_pipeline.md).

### Query pipeline

Process a new batch of images against the existing database:

```bash
# Set the batch name
export NEW_IMAGES_NAME="Batch1"

# Step 1: Segment and extract patterns from new images
python scripts/Q1-image_preparation.py
# → Review results in data/new/<batch_name>_checks/
#   Remove bad images from data/new/<batch_name>-pattern/

# Step 2: Match against DB and predict
python scripts/Q2-pattern_matching.py
python scripts/Q3-feature_aggregation.py
python scripts/Q4-model_application.py
# → Review results/top10_results_<batch_name>.csv
#   Confirm matches and assign IDs for new individuals

# Step 3: Add reviewed results to the main database
python scripts/Q5-add_results_to_db.py
```

For detailed information, see [README_predict_new_images.md](README_predict_new_images.md).

### SLURM (HPC)

SLURM wrapper scripts (`P-step*.sh`, `Q-step*.sh`) are provided for running on HPC clusters. For query scripts, pass the batch name as an argument:

```bash
sbatch Q-step1-image_preparation.sh Batch1
```

---

## Configuration

Key parameters are defined in `params/`:

- **`params/params.py`** — GlueStick and wireframe configuration, model paths, data paths.
- **`params/image_preparation_params.py`** — Segmentation model selection (`YOLO` or `GroundedSAM`), step toggles (`STEP_1A`, `STEP_1B`), image paths.

---

## Citation

If you use ReMatch in your research, please cite:

```
@article{alcaraz2025rematch,
  title={ReMatch: Re-identification of patterned species in open-set scenarios by matching keypoints and lines},
  author={Alcaraz, Roberto and Amores, Angel and Rotger, Andreu},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.