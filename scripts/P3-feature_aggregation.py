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

    if os.path.exists(params.PROCESSED_MATCHES_FILE_PATH):
        print("File already exists, check if you want to overwrite it.")
    else:
        # Open the LMDB environment in read-only mode
        env = lmdb.open(params.MATCHES_FILE_PATH)

        pairs, results = utils.read_pairs_and_results(env)
        processed_results = utils.process_results(pairs, results)

        # Create the DataFrame from the processed_results list
        matches = pd.DataFrame(
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

        matches = utils.remove_duplicate_images(matches)

        matches["id1"] = matches["img1_full"].map(lambda x: x.split("/")[0])
        matches["id2"] = matches["img2_full"].map(lambda x: x.split("/")[0])
        matches["same"] = matches["id1"] == matches["id2"]

        matches.reset_index(drop=True).to_parquet(
            params.PROCESSED_MATCHES_FILE_PATH, index=False
        )

    logging.info("-----------------------------------------------")
    logging.info("------- feature_aggregation.py finished! ------")
    logging.info("-----------------------------------------------")
