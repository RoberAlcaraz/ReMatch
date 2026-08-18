import os
import torch

NEW_IMAGES_NAME = os.environ["NEW_IMAGES_NAME"] if "NEW_IMAGES_NAME" in os.environ else "Batch1"  # Folder inside data/new with new images

#########################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rest of the paths
RAW_IMAGES_FOLDER = os.path.join("data", "images")
SEGMENTED_IMAGES_FOLDER = os.path.join("data", "images-segmented")
PATTERN_IMAGES_FOLDER = os.path.join("data", "images-pattern")
RESULTS_PATH = "results"

# New images to process
NEW_IMAGES_FOLDER = os.path.join("data", "new")
CHECK_RESULTS_PATH = f"{NEW_IMAGES_FOLDER}/{NEW_IMAGES_NAME}_checks"
NEW_RAW_IMAGES_FOLDER = os.path.join(NEW_IMAGES_FOLDER, NEW_IMAGES_NAME)
NEW_SEGMENTED_IMAGES_FOLDER = os.path.join(
    NEW_IMAGES_FOLDER, f"{NEW_IMAGES_NAME}-segmented"
)
NEW_PATTERN_IMAGES_FOLDER = os.path.join(
    NEW_IMAGES_FOLDER, f"{NEW_IMAGES_NAME}-pattern"
)


# Models
GROUNDING_DINO_CONFIG_PATH = "models/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "models/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"
YOLO_SEGMENTATION_MODEL = "models/yolo-segmentation.pt"
EDGE_NMS_PATH = "models/model.yml.gz"

# Parameters
STEP_1A = True  # Whether to perform segmentation
SEGMENTATION_MODEL = "YOLO"  # Options: "YOLO", "GroundedSAM"
STEP_1B = True  # Whether to extract patterns from segmented images

# --- Grounded-SAM settings (used when SEGMENTATION_MODEL == "GroundedSAM") ---
# CLASSES is the text prompt handed to GroundingDINO: name the animal, in the
# singular, as you would to a person ("lizard", "zebra", "seal", "shark").
# Grounded-SAM needs no per-species training, so it is the right thing to try
# first on a new species; YOLO is worth the annotation cost once you have a
# dataset large enough to fine-tune on.
CLASSES = ["lizard"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.7
# Detections below this confidence are kept but flagged in the log, so you can
# review them rather than silently losing images.
MIN_DETECTION_CONFIDENCE = 0.6

# --- Pattern extraction (STEP_1B) ---
# Isolates fine-scale texture with SAM + structured edge detection. Enabled here
# because the Balearic wall lizard is identified by its ventral scale mosaic; the
# other four datasets in the paper match on the segmented ROI directly and set
# this to False.
