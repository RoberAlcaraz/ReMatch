import os
import torch
from gluestick import GLUESTICK_ROOT

NEW_IMAGES_NAME = os.environ["NEW_IMAGES_NAME"] if "NEW_IMAGES_NAME" in os.environ else "Batch1"  # Folder inside data/new with new images

###########################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
