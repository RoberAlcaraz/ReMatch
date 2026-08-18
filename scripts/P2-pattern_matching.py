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

    device = params.DEVICE
    wireframe_conf = params.WIREFRAME_CONF
    image_height_resize = params.IMAGE_HEIGHT_RESIZE

    gluestick_conf = params.GLUESTICK_CONF

    unique_image_ids_path = params.UNIQUE_IMAGE_IDS_PATH
    wireframe_results_path = params.WIREFRAME_RESULTS_PATH
    matches_file_path = params.MATCHES_FILE_PATH

    utils.compute_wireframe(
        device,
        unique_image_ids_path,
        wireframe_results_path,
        wireframe_conf,
        image_height_resize,
        pattern_images_folder=params.PATTERN_IMAGES_FOLDER,
    )

    utils.pattern_matching(
        device,
        gluestick_conf,
        unique_image_ids_path,
        wireframe_results_path,
        matches_file_path,
    )

    logging.info("-----------------------------------------------")
    logging.info("-------- pattern_matching.py finished! --------")
    logging.info("-----------------------------------------------")
