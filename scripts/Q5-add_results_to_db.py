import sys
import pandas as pd
import os
import logging
from datetime import datetime
import subprocess

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


def save_image_to_folder(new_images_folder, dest_folder, img, dest_img):
    # Obtain the basename of the image
    img_basename = os.path.basename(img)
    img_path = os.path.join(new_images_folder, img_basename)

    dest_img_folder = os.path.join(dest_folder, dest_img.split("/")[0])
    dest_img_path = os.path.join(dest_folder, dest_img)
    os.makedirs(dest_img_folder, exist_ok=True)

    if "segmented" not in new_images_folder and "pattern" not in new_images_folder:
        dest_img_path = dest_img_path.replace(".png", ".jpg")
        img_path = img_path.replace(".png", ".jpg")

    if not os.path.exists(img_path):
        img_path = img_path.replace(".jpg", ".JPG")

    if os.path.exists(dest_img_path):
        logging.info(f"Image {dest_img_path} already exists.")
        return img_path, dest_img_path

    # Copy the image to the new folder
    try:
        subprocess.run(
            ["cp", img_path, dest_img_path], check=True, stdout=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        subprocess.run(
            ["sudo", "cp", img_path, dest_img_path], check=True, stdout=subprocess.PIPE
        )
    logging.info(f"Image {img_path} copied to {dest_img_path}")
    return img_path, dest_img_path


def merge_matches(matches, new_matches, results):
    logging.info(
        "Create a new column with the image name by replacing the path by the column pred_lizard"
    )
    results["new_img"] = results.apply(
        lambda row: (
            row["img1_full"].replace(row["img1_full"].split("/")[0], str(row["id1"]))
            if pd.notna(row["id1"])
            else pd.NA
        ),
        axis=1,
    )
    # Keep only the two columns we need and drop missing rows
    results = results[["img1_full", "new_img"]].dropna()
    # Deduplicate the mapping so the same image isn't processed multiple times
    results = results.drop_duplicates().reset_index(drop=True)

    logging.info("Adding the results to the new matches dataframe")
    new_matches = new_matches.merge(
        results, left_on="img1_full", right_on="img1_full", how="left"
    )
    new_matches = new_matches.drop(columns=["img1_full"])
    new_matches = new_matches.rename(columns={"new_img": "img1_full"})
    new_matches = new_matches[
        [
            "img1_full",
            "img2_full",
            "num_nonzero_points",
            "mean_point_prob",
            "num_nonzero_lines",
            "mean_line_prob",
        ]
    ]
    new_matches = new_matches.dropna().reset_index(drop=True)

    print(new_matches)
    new_matches["id1"] = new_matches["img1_full"].apply(lambda x: x.split("/")[0])
    new_matches["id2"] = new_matches["img2_full"].apply(lambda x: x.split("/")[0])

    logging.info("Concat the matches and new matches dataframes")
    matches = pd.concat([matches, new_matches], ignore_index=True)
    matches = matches[~matches["img1_full"].str.contains("pattern")].reset_index(
        drop=True
    )
    matches = matches.drop_duplicates(ignore_index=True)
    return matches, results


def save_images_to_directories(
    new_images_folder,
    images_folder,
    new_segmented_images_folder,
    segmented_images_folder,
    new_pattern_images_folder,
    pattern_images_folder,
    results,
):
    # Ensure we only process unique (img, new_img) pairs to avoid duplicate copies
    pairs = results[["img1_full", "new_img"]].drop_duplicates().values
    for img, new_img in pairs:
        save_image_to_folder(new_images_folder, images_folder, img, new_img)
        save_image_to_folder(
            new_segmented_images_folder, segmented_images_folder, img, new_img
        )
        save_image_to_folder(
            new_pattern_images_folder, pattern_images_folder, img, new_img
        )


def add_results_to_db(
    unique_image_ids_path,
    new_unique_image_ids_path,
    wireframe_results_path,
    new_wireframe_results_path,
    matches_processed_file_path,
    new_matches_processed_file_path,
    top10_results_path,
    new_images_folder,
    images_folder,
    new_segmented_images_folder,
    segmented_images_folder,
    new_pattern_images_folder,
    pattern_images_folder,
    device,
):

    logging.info("Loading the image ids...")
    unique_image_ids = utils.read_unique_images_ids(unique_image_ids_path)
    new_unique_image_ids = utils.read_unique_images_ids(new_unique_image_ids_path)

    logging.info("Loading new precomputed wireframes...")
    print(new_wireframe_results_path)
    new_wireframe = utils.load_wireframe(
        device,
        new_wireframe_results_path,
        new_unique_image_ids,
    )

    logging.info("Loading the cleaned matches data...")
    matches = pd.read_parquet(matches_processed_file_path)
    matches = matches.reset_index(drop=True)
    matches["id1"] = matches["img1_full"].map(lambda x: x.split("/")[0])
    matches["id2"] = matches["img2_full"].map(lambda x: x.split("/")[0])
    matches["same"] = matches["id1"] == matches["id2"]
    print("Length of matches: ", len(matches))
    print(matches)

    logging.info("Loading the NEW cleaned matches data...")
    new_matches = pd.read_parquet(new_matches_processed_file_path)
    new_matches = new_matches.reset_index(drop=True)
    new_matches["id1"] = new_matches["img1_full"].map(lambda x: x.split("/")[0])
    new_matches["id2"] = new_matches["img2_full"].map(lambda x: x.split("/")[0])
    print("Length of new matches: ", len(new_matches))

    logging.info("Loading results data...")
    results = pd.read_csv(top10_results_path)

    logging.info("Merging the matches with the new matches...")
    matches, results = merge_matches(matches, new_matches, results)
    print(matches)
    logging.info("Saving the matches dataframe with the new results...")
    matches.to_parquet(matches_processed_file_path, index=False)
    logging.info("Matches saved successfully.")

    logging.info("Renaming the keys of the new precomputed features...")
    img1_to_new_img = results.groupby("img1_full")["new_img"].first()
    new_mapping = {k: img1_to_new_img.get(k, k) for k in new_wireframe.keys()}
    new_wireframe = {new_mapping[k]: v for k, v in new_wireframe.items()}

    logging.info("Renaming the keys of the new precomputed features...")
    utils.rename_h5_keys(new_wireframe_results_path, new_mapping)

    logging.info("Saving the new precomputed features...")
    utils.add_new_wireframe(
        device=device,
        wireframe_path=wireframe_results_path,
        new_wireframe=new_wireframe,
        new_unique_image_ids=list(new_wireframe.keys()),
    )
    logging.info("New precomputed features saved successfully.")

    logging.info("Adding the new ids to the unique ids file...")
    with open(unique_image_ids_path, "a") as f:
        for img_id in new_wireframe.keys():
            if img_id not in unique_image_ids:
                f.write(f"{img_id}\n")
    logging.info("New ids added successfully.")

    logging.info("Creating new folders for the new images...")
    save_images_to_directories(
        new_images_folder,
        images_folder,
        new_segmented_images_folder,
        segmented_images_folder,
        new_pattern_images_folder,
        pattern_images_folder,
        results,
    )
    logging.info("Folders created successfully.")


if __name__ == "__main__":
    logging.info("-----------------------------------------------")
    logging.info("----- Running add_results_to_database.py  -----")
    logging.info("-----------------------------------------------")

    device = params.DEVICE
    new_images_name = params.NEW_IMAGES_NAME
    new_images_folder = params.NEW_IMAGES_FOLDER
    images_folder = params.RAW_IMAGES_FOLDER
    new_segmented_images_folder = params.NEW_SEGMENTED_IMAGES_FOLDER
    segmented_images_folder = params.SEGMENTED_IMAGES_FOLDER
    # With STEP_1B off these are the segmented folders, not the pattern ones -
    # the batch has to be merged into whatever the database actually matches on.
    new_pattern_images_folder = params.NEW_MATCHING_IMAGES_FOLDER
    pattern_images_folder = params.MATCHING_IMAGES_FOLDER
    unique_image_ids_path = params.UNIQUE_IMAGE_IDS_PATH
    new_unique_image_ids_path = params.NEW_UNIQUE_IMAGE_IDS_PATH
    wireframe_results_path = params.WIREFRAME_RESULTS_PATH
    new_wireframe_results_path = params.NEW_WIREFRAME_RESULTS_PATH
    processed_matches_file_path = params.PROCESSED_MATCHES_FILE_PATH
    new_processed_matches_file_path = params.NEW_PROCESSED_MATCHES_FILE_PATH
    top10_results_path = params.TOP10_RESULTS_PATH

    add_results_to_db(
        unique_image_ids_path,
        new_unique_image_ids_path,
        wireframe_results_path,
        new_wireframe_results_path,
        processed_matches_file_path,
        new_processed_matches_file_path,
        top10_results_path,
        new_images_folder,
        images_folder,
        new_segmented_images_folder,
        segmented_images_folder,
        new_pattern_images_folder,
        pattern_images_folder,
        device,
    )
