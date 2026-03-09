import os
import logging
import sys
import random
import glob

# from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import params.image_preparation_params as params
import utils.image_preparation_utils as utils


# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Stream to stdout
    ],
    force=True,
)


if __name__ == "__main__":
    logging.info("-----------------------------------------------")
    logging.info("------ Running 1. IMAGE PREPARATION...  -------")
    logging.info("-----------------------------------------------")

    # Define parameters
    # These parameters are defined in params/image_preparation_params.py
    # They include paths for raw images, segmented images, pattern images, results,
    # and model configurations.
    # Ensure these paths are correctly set in the params file.
    # If you need to change the paths, do it in params/image_preparation_params.py
    device = params.DEVICE

    raw_images_folder = params.NEW_RAW_IMAGES_FOLDER
    segmented_images_folder = params.NEW_SEGMENTED_IMAGES_FOLDER
    pattern_images_folder = params.NEW_PATTERN_IMAGES_FOLDER

    check_results_path = params.CHECK_RESULTS_PATH
    yolo_model_path = params.YOLO_SEGMENTATION_MODEL
    sam_checkpoint_path = params.SAM_CHECKPOINT_PATH
    edge_nms_path = params.EDGE_NMS_PATH

    step_1a = params.STEP_1A
    step_1b = params.STEP_1B
    segmentation_model = params.SEGMENTATION_MODEL

    # STEP 1a: Segment images using GroundedSAM or YOLO
    if step_1a:
        logging.info("STEP 1a: Segmenting images...")
        os.makedirs(segmented_images_folder, exist_ok=True)
        os.makedirs(check_results_path, exist_ok=True)
        # Using GroundedSAM:
        # if segmentation_model == "GroundedSAM":
        #     logging.info("Using GroundedSAM for segmentation...")
        #     # Building GroundingDINO inference model
        #     grounding_dino_model = Model(
        #         model_config_path=params.GROUNDING_DINO_CONFIG_PATH,
        #         model_checkpoint_path=params.GROUNDING_DINO_CHECKPOINT_PATH,
        #     )
        #
        #     # Building SAM Model and SAM Predictor
        #     sam = sam_model_registry[params.SAM_ENCODER_VERSION](
        #         checkpoint=params.SAM_CHECKPOINT_PATH
        #     )
        #     sam.to(device=params.DEVICE)
        #     sam_predictor = SamPredictor(sam)
        #     utils.GSAM_segmentation(grounding_dino_model, sam_predictor, classes)

        if segmentation_model == "YOLO":
            logging.info("Using YOLO for segmentation...")
            utils.YOLO_segmentation(
                yolo_model_path,
                raw_images_folder,
                segmented_images_folder,
                check_results_path,
            )

        else:
            raise ValueError("Invalid segmentation model specified.")

    # Using YOLO:
    # utils.YOLO_segmentation(yolo_model_path, raw_images_folder, segmented_images_folder)

    # STEP 1b: Prepare images for pattern matching
    if step_1b:
        logging.info("STEP 1b: Extracting patterns from segmented images...")
        os.makedirs(pattern_images_folder, exist_ok=True)
        os.makedirs(check_results_path, exist_ok=True)

        utils.extract_pattern_from_images(
            segmented_images_folder,
            pattern_images_folder,
            check_results_path,
            sam_checkpoint_path,
            edge_nms_path,
            device=device,
        )

    """
    # STEP 1c: Divide images into train and test sets and save image paths
    logging.info("STEP 1c: Dividing images into train and test sets...")
    os.makedirs(good_images_folder, exist_ok=True)
    random.seed(42)

    if step_1b:
        # If patterns were extracted, use the pattern images folder
        utils.split_dataset(
            pattern_images_folder, train_folder, test_folder, p_unseen, p_train_seen
        )
    else:
        # If no patterns were extracted, use the segmented images folder
        logging.info(
            "No patterns extracted. Using segmented images for train/test split."
        )
        # This assumes that the segmented images are used for training/testing.
        # If you want to use raw images instead, change the folder accordingly.
        utils.split_dataset(
            segmented_images_folder, train_folder, test_folder, p_unseen, p_train_seen
        )
    # Group test images randomly in folders of 10 images each
    utils.group_test_images(test_folder, group_size=20)

    # Iterate through group folders
    utils.save_image_paths(train_folder, test_folder, good_images_folder)

    # Save metadata for the train and test folders
    utils.save_metadata(train_folder, test_folder)
    """

    # Save img paths to a text file
    # pattern_img_paths = glob.glob(os.path.join(pattern_images_folder, "*.png"))
    # with open(
    #     os.path.join(results_path, f"unique_ids_{params.NEW_IMAGES_NAME}.txt"), "w"
    # ) as f:
    #     for path in pattern_img_paths:
    #         path = path.replace("data/new/", "")
    #         f.write(f"{path}\n")

    logging.info("-----------------------------------------------")
    logging.info("-------- IMAGE PREPARATION finished! ----------")
    logging.info("-----------------------------------------------")
