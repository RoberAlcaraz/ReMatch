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
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"
YOLO_SEGMENTATION_MODEL = "models/yolo-segmentation.pt"
EDGE_NMS_PATH = "models/model.yml.gz"

# Parameters
STEP_1A = True  # Whether to perform segmentation
SEGMENTATION_MODEL = "YOLO"  # Options: "GroundedSAM", "YOLO"
STEP_1B = True  # Whether to extract patterns from segmented images
