import logging
import sys

import utils.utils as utils
import params.params as params

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
    logging.info("------- Running pattern_matching.py...  -------")
    logging.info("-----------------------------------------------")

    utils.set_seed(params.RANDOM_SEED)

    device = params.DEVICE
    wireframe_conf = params.WIREFRAME_CONF
    image_height_resize = params.IMAGE_HEIGHT_RESIZE

    gluestick_conf = params.GLUESTICK_CONF

    unique_image_ids_path = params.UNIQUE_IMAGE_IDS_PATH
    wireframe_results_path = params.WIREFRAME_RESULTS_PATH
    matches_file_path = params.MATCHES_FILE_PATH

    new_pattern_images_folder = params.NEW_MATCHING_IMAGES_FOLDER
    new_unique_image_ids_path = params.NEW_UNIQUE_IMAGE_IDS_PATH
    new_wireframe_results_path = params.NEW_WIREFRAME_RESULTS_PATH
    new_matches_file_path = params.NEW_MATCHES_FILE_PATH

    # Step 1: Compute wireframes for the unique images
    utils.compute_wireframe(
        device,
        unique_image_ids_path,
        wireframe_results_path,
        wireframe_conf,
        image_height_resize,
        new_unique_image_ids_path,
        new_pattern_images_folder,
        new_wireframe_results_path,
    )

    # Step 2: Perform pattern matching using Gluestick
    utils.pattern_matching(
        device,
        gluestick_conf,
        unique_image_ids_path,
        wireframe_results_path,
        matches_file_path,
        new_unique_image_ids_path,
        new_wireframe_results_path,
        new_matches_file_path,
    )

    logging.info("-----------------------------------------------")
    logging.info("-------- pattern_matching.py finished! --------")
    logging.info("-----------------------------------------------")
