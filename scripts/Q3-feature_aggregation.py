import os
import pandas as pd
import lmdb
import logging
import sys

import params.params as params
import utils.utils as utils

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
    logging.info("-------- Running feature_aggregation.py -------")
    logging.info("-----------------------------------------------")

    if os.path.exists(params.NEW_PROCESSED_MATCHES_FILE_PATH):
        print("File already exists, check if you want to overwrite it.")
    else:
        # Open the LMDB environment in read-only mode
        env = lmdb.open(params.NEW_MATCHES_FILE_PATH)

        pairs, results = utils.read_pairs_and_results(env)
        processed_results = utils.process_results(pairs, results)

        # Create the DataFrame from the processed_results list
        new_matches = pd.DataFrame(
            processed_results,
            columns=[
                "img1_full",
                "img2_full",
                "num_nonzero_points",
                "mean_point_prob",
                "num_nonzero_lines",
                "mean_line_prob",
            ],
        )
        new_matches = utils.remove_duplicate_images(new_matches)

        new_matches["id1"] = new_matches["img1_full"].map(lambda x: x.split("/")[0])
        new_matches["id2"] = new_matches["img2_full"].map(lambda x: x.split("/")[0])
        new_matches["same"] = new_matches["id1"] == new_matches["id2"]

        new_matches.reset_index(drop=True).to_parquet(
            params.NEW_PROCESSED_MATCHES_FILE_PATH, index=False
        )

    logging.info("-----------------------------------------------")
    logging.info("------- feature_aggregation.py finished! ------")
    logging.info("-----------------------------------------------")
