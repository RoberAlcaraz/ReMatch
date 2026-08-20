import os
import torch
from gluestick import GLUESTICK_ROOT
from params.image_preparation_params import STEP_1B

NEW_IMAGES_NAME = os.environ["NEW_IMAGES_NAME"] if "NEW_IMAGES_NAME" in os.environ else "Batch1"  # Folder inside data/new with new images

###########################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for the random draws inside the pipeline (SuperPoint's keypoint padding,
# the Grounded-SAM mask-centre check). Set to None to leave them unseeded.
RANDOM_SEED = 0

# Two photographs matching above this many keypoints are treated as the same
# image in the gallery twice, and one copy is dropped. Two genuinely different
# photographs of one animal share a few hundred points at most, so the gap is
# wide - but it is a heuristic, and the demo notebooks show you every pair it
# flags so you can judge before it acts.
DUPLICATE_POINT_THRESHOLD = 900

# Models
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"


# Configuration for SPWireframeDescriptor
WIREFRAME_CONF = {
    "sp_params": {
        "force_num_keypoints": True,
        "max_num_keypoints": 1000,
    },
    "wireframe_params": {
        "merge_points": True,
        "merge_line_endpoints": True,
    },
    "max_n_lines": 300,
}
IMAGE_HEIGHT_RESIZE = 670


# Configuration for GlueStick
GLUESTICK_CONF = {
    "name": "two_view_pipeline",
    "use_lines": True,
    "matcher": {
        "name": "gluestick",
        "weights": str(
            GLUESTICK_ROOT / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
        ),
        "trainable": False,
    },
    "ground_truth": {
        "from_pose_depth": False,
    },
}

# Paths
DATABASE_PATH = os.path.join(
    "data", "metadata.csv"
)  # identity, image_id, path format (C1160,IMG_7936,train/C1160/IMG_7936.png, for example)
RAW_IMAGES_FOLDER = os.path.join("data", "images")
SEGMENTED_IMAGES_FOLDER = os.path.join("data", "images-segmented")
PATTERN_IMAGES_FOLDER = os.path.join("data", "images-pattern")

# Pattern extraction (STEP_1B in params/image_preparation_params.py) is
# optional. Lizards need it: the identity is in the ventral scale mosaic, so a
# second pass isolates that texture into images-pattern/. Zebras, seals and
# whale sharks do not — the stripes, rings and spots *are* the pattern, and
# matching runs on the segmented ROI directly. MATCHING_IMAGES_FOLDER is
# whichever of the two the rest of the pipeline should read.
MATCHING_IMAGES_FOLDER = PATTERN_IMAGES_FOLDER if STEP_1B else SEGMENTED_IMAGES_FOLDER
DATABASES_FOLDER = "databases"
RESULTS_FOLDER = "results"
UNIQUE_IMAGE_IDS_PATH = os.path.join(RESULTS_FOLDER, "unique_ids.txt")
GOOD_IMAGES_PATH = os.path.join(RESULTS_FOLDER, "good_images", "train_good_images.log")
WIREFRAME_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "precomputed_wireframe.h5")
MATCHES_FILE_PATH = os.path.join(RESULTS_FOLDER, "matches.lmdb")
PROCESSED_MATCHES_FILE_PATH = os.path.join(RESULTS_FOLDER, "processed_matches.parquet")
BEST_MODEL_PATH = os.path.join(RESULTS_FOLDER, "best_classification_model.pkl")
SCALER_PATH = os.path.join(RESULTS_FOLDER, "scaler.pkl")
LOGREG_MODEL_PATH = os.path.join(RESULTS_FOLDER, "logistic_regression_model.pkl")
BEST_THRESHOLD_PATH = os.path.join(RESULTS_FOLDER, "threshold.txt")

# New paths for updated results
NEW_IMAGES_FOLDER = os.path.join("data", "new", f"{NEW_IMAGES_NAME}")
NEW_SEGMENTED_IMAGES_FOLDER = os.path.join(
    "data", "new", f"{NEW_IMAGES_NAME}-segmented"
)
NEW_PATTERN_IMAGES_FOLDER = os.path.join("data", "new", f"{NEW_IMAGES_NAME}-pattern")
NEW_MATCHING_IMAGES_FOLDER = (
    NEW_PATTERN_IMAGES_FOLDER if STEP_1B else NEW_SEGMENTED_IMAGES_FOLDER
)
NEW_UNIQUE_IMAGE_IDS_PATH = os.path.join(
    RESULTS_FOLDER, f"unique_ids_{NEW_IMAGES_NAME}.txt"
)
NEW_WIREFRAME_RESULTS_PATH = os.path.join(
    RESULTS_FOLDER, f"precomputed_wireframe_{NEW_IMAGES_NAME}.h5"
)
NEW_PRECOMPUTED_FEATURES_PATH = os.path.join(
    RESULTS_FOLDER, f"precomputed_features_{NEW_IMAGES_NAME}.h5"
)
NEW_MATCHES_FILE_PATH = os.path.join(RESULTS_FOLDER, f"matches_{NEW_IMAGES_NAME}.lmdb")
NEW_PROCESSED_MATCHES_FILE_PATH = os.path.join(
    RESULTS_FOLDER, f"processed_matches_{NEW_IMAGES_NAME}.parquet"
)
TOP10_RESULTS_PATH = os.path.join(
    RESULTS_FOLDER, f"top10_results_{NEW_IMAGES_NAME}.csv"
)
