import glob
import os
import sys
import torch
import h5py
import pickle
import numpy as np
import pandas as pd
import cv2
import logging
from tqdm import tqdm
import lmdb

from gluestick.models.wireframe import SPWireframeDescriptor
from gluestick import numpy_image_to_torch
from gluestick.models.two_view_pipeline_precomputed_wireframe import (
    TwoViewPipeline,
)


def set_seed(seed=0):
    """Make a run reproducible.

    Two steps of the pipeline draw random numbers. SuperPoint runs with
    ``force_num_keypoints``, so when it detects fewer than
    ``max_num_keypoints`` it pads the rest with uniformly random points; on the
    bundled lizard images that is about 16% of every image's keypoints. And the
    Grounded-SAM step samples the centre of each mask at random to decide
    whether SAM returned the animal or the background.

    Left unseeded, two runs of the same data select different pair-level
    classifiers and land on visibly different thresholds. Seeding does not
    change the method, it just pins which draw you get.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_transparent_img(image_path):
    """Read a segmented or pattern image as grayscale, with the background black.

    Handles the three forms these images take on disk: RGBA (the usual output of
    the segmentation step, where alpha carries the mask), plain BGR, and
    single-channel grayscale (pattern crops are often already grayscale).
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 2:  # already grayscale, nothing to composite
        return image

    if image.shape[2] == 4:
        # Blacken everything outside the mask, then drop the alpha channel.
        alpha = image[:, :, 3]
        image = image[:, :, :3].copy()
        image[alpha == 0] = 0

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def read_unique_images_ids(unique_image_ids_path):
    try:
        with open(unique_image_ids_path, "r") as f:
            unique_image_ids = f.readlines()
            unique_image_ids = [img_id.strip() for img_id in unique_image_ids]
        logging.info("Successfully loaded unique image IDs.")
    except Exception as e:
        logging.error(f"Error loading unique image IDs: {e}")
        raise
    return unique_image_ids


# Resize the images to a smaller size but maintain the aspect ratio
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        aspect_ratio = height / float(h)
        dim = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def enhance_contrast(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_wireframe(
    device,
    unique_image_ids_path,
    wireframe_results_path,
    wireframe_conf,
    image_height_resize,
    new_unique_image_ids_path=None,
    new_pattern_images_folder=None,
    new_wireframe_results_path=None,
    pattern_images_folder=os.path.join("data", "images-pattern"),
):
    """Compute (or resume) the per-image SuperPoint/LSD wireframe cache.

    Ids in ``unique_image_ids_path`` are stored relative to
    ``pattern_images_folder`` (training) or to the parent of
    ``new_pattern_images_folder`` (query), so that the same id keys the wireframe
    HDF5 cache, the LMDB match store and the feature tables. Pass
    ``pattern_images_folder`` explicitly if you changed the corresponding path in
    ``params/params.py``.
    """
    
    # Some clusters sit behind a TLS-intercepting proxy, which breaks the
    # one-time download of the SuperPoint / GlueStick weights. Disabling
    # certificate verification is an opt-in escape hatch, never the default:
    # it applies process-wide and removes protection against tampering.
    if os.environ.get("REMATCH_INSECURE_SSL") == "1":
        import ssl

        logging.warning(
            "REMATCH_INSECURE_SSL=1: TLS certificate verification is disabled "
            "for this process. Prefer fixing the trust store, or download the "
            "weights manually into resources/weights/."
        )
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

    wireframe = SPWireframeDescriptor(wireframe_conf).to(device)

    if new_pattern_images_folder is None:
        unique_ids = pd.read_csv(unique_image_ids_path, header=None, names=["img"])
        unique_ids = unique_ids["img"].to_list()
        logging.info(f"Found {len(unique_ids)} images to process.")
        img_paths = [
            os.path.join(pattern_images_folder, img_id) for img_id in unique_ids
        ]
        # unique_image_ids = []
        # for img_path in unique_ids:
        #     # Get the image name and the last folder name
        #     img_name = os.path.basename(img_path)
        #     dir_name = os.path.basename(os.path.dirname(img_path))
        #     unique_image_ids.append(f"{dir_name}/{img_name}")

        # Save the unique image ids
        # with open(unique_image_ids_path, "w") as f:
        #     for img_id in unique_image_ids:
        #         f.write(f"{img_id}\n")
    else:
        # unique_ids = pd.read_csv(new_unique_image_ids_path, header=None, names=["img"])
        # unique_ids = unique_ids["img"].to_list()
        # print(new_pattern_images_folder)
        img_paths = sorted(
            glob.glob(os.path.join(new_pattern_images_folder, "*.png"))
            + glob.glob(os.path.join(new_pattern_images_folder, "*.jpg"))
            + glob.glob(os.path.join(new_pattern_images_folder, "*.JPG"))
        )
        # Ids are stored relative to the batch's parent directory (data/new),
        # so they read as "<batch>-pattern/<image>.png".
        new_images_root = os.path.dirname(new_pattern_images_folder.rstrip(os.sep))
        unique_ids = [os.path.relpath(x, new_images_root) for x in img_paths]
        logging.info(f"Found {len(unique_ids)} images to process.")
        # img_paths = [os.path.join("data", "new", img_id) for img_id in unique_ids]

        # Save the NEW unique image ids
        with open(new_unique_image_ids_path, "w") as f:
            for img_id in unique_ids:
                f.write(f"{img_id}\n")

    logging.info("Computing wireframes...")
    if new_wireframe_results_path is not None:
        wireframe_results_path = new_wireframe_results_path

    # Open the HDF5 file for saving features
    if os.path.exists(wireframe_results_path):
        logging.info(
            f"Path {wireframe_results_path} already exists. Check it if needed. Continuing the matching process..."
        )
    else:
        with h5py.File(wireframe_results_path, "a") as hdf5_file:
            for img_path in tqdm(img_paths):
                # print(f"Processing {img_path}...")

                # Get the image name and the last folder name
                img_name = os.path.basename(img_path)
                dir_name = os.path.basename(os.path.dirname(img_path))
                image_id = f"{dir_name}/{img_name}"
                print(f"Image ID: {image_id}")
                if not os.path.exists(img_path):
                    logging.warning(f"Image {img_path} does not exist. Skipping.")
                    continue

                # Check if the image data already exists in the HDF5 file
                if image_id in hdf5_file:
                    # logging.info(f"Skipping {image_id}, already processed.")
                    continue

                # Prepare data dictionary for passing into models
                # read_transparent_img, not cv2.IMREAD_GRAYSCALE: segmented
                # images carry their mask in the alpha channel, and a plain
                # grayscale read discards it - leaving the background in the
                # image and undoing the segmentation. Only matters for RGBA
                # inputs, which is every species that skips pattern extraction.
                image = read_transparent_img(img_path)
                image = resize_image(image, height=image_height_resize)
                image = enhance_contrast(image)
                img_tensor = numpy_image_to_torch(image).to(device)[None]
                data = {"image": img_tensor}

                # Step 3: Run Wireframe model (it has internal SuperPoint model)
                # logging.info(f"Processing wireframe for {image_id}...")
                with torch.no_grad():
                    wireframe_result = wireframe._forward(
                        data, save_path=wireframe_results_path, image_id=image_id
                    )

                del image, img_tensor, data, wireframe_result
                torch.cuda.empty_cache()
        logging.info(
            f"Feature computation and saving complete for {len(img_paths)} images."
        )


def load_wireframe(device, wireframe_path, unique_image_ids):
    logging.info("Loading precomputed features from HDF5...")
    wireframe = {}
    if os.path.exists(wireframe_path):
        try:
            with h5py.File(wireframe_path, "r") as hdf5_file:
                for image_id in unique_image_ids:
                    # print(f"Loading features for {image_id}...")
                    grp = hdf5_file[image_id]
                    wireframe[image_id] = {
                        "image_size": torch.Size(grp["image_size"][:]),
                        # "image": torch.tensor(grp["image"][:]).to(device),
                        "keypoints": torch.tensor(grp["keypoints"][:]).to(device),
                        "keypoint_scores": torch.tensor(grp["keypoint_scores"][:]).to(
                            device
                        ),
                        "descriptors": torch.tensor(grp["descriptors"][:]).to(device),
                        "lines": torch.tensor(grp["lines"][:]).to(device),
                        "line_scores": torch.tensor(grp["line_scores"][:]).to(device),
                        "pl_associativity": torch.tensor(grp["pl_associativity"][:]).to(
                            device
                        ),
                        "lines_junc_idx": torch.tensor(grp["lines_junc_idx"][:]).to(
                            device
                        ),
                    }
            logging.info("Loaded all precomputed features from HDF5.")
        except Exception as e:

            logging.error(f"Error loading precomputed features for: {e}")
            raise
    else:
        logging.error(f"Precomputed features file not found: {wireframe_path}")
        raise FileNotFoundError(
            f"Precomputed features file not found: {wireframe_path}"
        )

    return wireframe


def compute_point_and_line_matches(
    pipeline,
    results,
    img_id0,
    img_id1,
    wireframe,
    new_wireframe=None,
):
    if new_wireframe is not None:
        data = {
            "image_size0": new_wireframe[img_id0]["image_size"],
            "image_size1": wireframe[img_id1]["image_size"],
        }
        pred0 = new_wireframe[img_id0].copy()
    else:
        data = {
            "image_size0": wireframe[img_id0]["image_size"],
            "image_size1": wireframe[img_id1]["image_size"],
        }
        pred0 = wireframe[img_id0].copy()

    pred1 = wireframe[img_id1].copy()

    pred = {
        **{k + "0": v for k, v in pred0.items()},
        **{k + "1": v for k, v in pred1.items()},
    }

    # Run the TwoViewPipeline
    with torch.no_grad():
        match_result = pipeline._forward(data, pred)

    results.append(
        (
            img_id0,
            img_id1,
            match_result["match_scores0"],
            match_result["line_match_scores0"],
        )
    )


def write_results_to_lmdb(results, lmdb_env):
    """Save the results to LMDB."""
    with lmdb_env.begin(write=True) as txn:
        for img1, img2, point_probs, line_probs in results:
            pair_key = f"pair_{img1}_{img2}".encode()
            value = pickle.dumps((point_probs.cpu().numpy(), line_probs.cpu().numpy()))
            txn.put(pair_key, value)

    # Clear results after writing to save memory
    results.clear()


def pattern_matching(
    device,
    gluestick_conf,
    unique_image_ids_path,
    wireframe_path,
    matches_file_path,
    new_unique_image_ids_path=None,
    new_wireframe_results_path=None,
    new_matches_file_path=None,
):
    logging.info(f"Using device: {device}")
    pipeline = TwoViewPipeline(gluestick_conf).to(device).eval()

    unique_image_ids = read_unique_images_ids(unique_image_ids_path)
    logging.info(f"Total unique images: {len(unique_image_ids)}")
    if new_unique_image_ids_path is not None:
        new_unique_image_ids = read_unique_images_ids(new_unique_image_ids_path)
        logging.info(f"New unique images: {len(new_unique_image_ids)}")

        # Get the image pairs between the new images and the images in the DB
        image_pairs = [
            (img_id0, img_id1)
            for img_id0 in new_unique_image_ids
            for img_id1 in unique_image_ids
        ]
        new_wireframe = load_wireframe(
            device, new_wireframe_results_path, new_unique_image_ids
        )
        logging.info(f"Total image pairs: {len(image_pairs)}")
    else:
        image_pairs = [
            (img1, img2)
            for i, img1 in enumerate(unique_image_ids)
            for img2 in unique_image_ids[i + 1 :]
        ]
        logging.info(f"Total image pairs: {len(image_pairs)}")
        new_wireframe = None

    wireframe = load_wireframe(device, wireframe_path, unique_image_ids)

    results = []
    total_pairs = len(image_pairs)
    percent_interval = max(1, total_pairs // 100)

    if new_matches_file_path is not None:
        lmdb_env = lmdb.open(
            new_matches_file_path,
            map_size=int(
                1099511627776 / 2
            ),  # Set an appropriate map size (500GB here, adjust if necessary)
            create=True,  # Create the LMDB file if it doesn't exist
            writemap=True,  # Use memory-mapped I/O, which is faster for writing large amounts of data
            map_async=True,  # Allow the environment to asynchronously flush the data to disk
        )
    else:
        lmdb_env = lmdb.open(
            matches_file_path,
            map_size=int(
                1099511627776 / 2
            ),  # Set an appropriate map size (500GB here, adjust if necessary)
            create=True,  # Create the LMDB file if it doesn't exist
            writemap=True,  # Use memory-mapped I/O, which is faster for writing large amounts of data
            map_async=True,  # Allow the environment to asynchronously flush the data to disk
        )

    for idx, (img_id0, img_id1) in tqdm(enumerate(image_pairs), total=total_pairs):
        # print(f"Processing pair: {img_id0}, {img_id1}")
        if idx % percent_interval == 0:
            logging.info(f"{(idx / total_pairs) * 100:.2f}% completed")

        # Check if the pair exists in the LMDB database
        with lmdb_env.begin(write=False) as txn:
            if txn.get(f"pair_{img_id0}_{img_id1}".encode()) is not None:
                # logging.info(f"Pair {img_id0}, {img_id1} already exists, skipping.")
                continue

        # Compute matches
        compute_point_and_line_matches(
            pipeline,
            results,
            img_id0,
            img_id1,
            wireframe,
            new_wireframe,
        )

        if (idx + 1) % 10000 == 0:
            logging.info("Writing 10,000 results to LMDB...")
            write_results_to_lmdb(results, lmdb_env)

    if results:
        logging.info("Writing final results to LMDB...")
        write_results_to_lmdb(results, lmdb_env)

    logging.info("Matching process completed.")


def read_pairs_and_results(env):
    pairs = []
    results = []

    # Start a transaction to read all the data
    with env.begin() as txn:
        cursor = txn.cursor()  # Create a cursor to iterate through the database

        # Iterate over all key-value pairs in the database
        for key, value in tqdm(cursor):
            # Deserialize the key
            # print(key.decode("utf-8").replace("pair_", "").split(".png_"))
            ext = key.decode("utf-8").split(".")[-1]
            img1, img2 = key.decode("utf-8").replace("pair_", "").split(f".{ext}_")
            img1 = img1 + f".{ext}"
            pairs.append((img1, img2))
            results.append((pickle.loads(value)))
    return pairs, results


def process_results(pairs, results):
    processed_results = []

    # When the results are read, they are sorted,
    # So we can iterate over the image pairs and results in parallel
    for image_pair, result in tqdm(zip(pairs, results)):
        img1_full, img2_full = image_pair
        point_probs, line_probs = result

        # Calculate the number of non-zero points and lines
        num_nonzero_points = (
            (point_probs > 0.2).sum().item()
        )  # Count of points with probability > 0
        num_nonzero_lines = (
            (line_probs > 0.2).sum().item()
        )  # Count of lines with probability > 0

        # Calculate the mean probability, ignoring zeros
        if num_nonzero_points > 0:
            mean_point_prob = point_probs[point_probs > 0.2].mean().item()
        else:
            mean_point_prob = 0.0  # If no points > 0, set mean to 0

        if num_nonzero_lines > 0:
            mean_line_prob = line_probs[line_probs > 0.2].mean().item()
        else:
            mean_line_prob = 0.0  # If no lines > 0, set mean to 0

        # Append the new metrics along with the original information
        processed_results.append(
            (
                img1_full,
                img2_full,
                num_nonzero_points,
                mean_point_prob,
                num_nonzero_lines,
                mean_line_prob,
            )
        )

    return processed_results


def remove_duplicate_images(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate images from the matches DataFrame.

    Args:
        matches (pd.DataFrame): DataFrame containing match data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    print("Removing duplicate images...")
    from params.params import DUPLICATE_POINT_THRESHOLD

    same_images = matches[matches["num_nonzero_points"] > DUPLICATE_POINT_THRESHOLD]
    print(same_images)
    same_images_index = same_images.index
    same_images_path = same_images.img1_full.unique()
    print(same_images_path)

    # Remove rows with duplicate images based on index
    matches = matches.loc[~matches.index.isin(same_images_index)].reset_index(drop=True)

    # Remove rows where img1_full or img2_full are in the duplicate paths
    matches = matches.loc[
        ~matches["img1_full"].isin(same_images_path)
        & ~matches["img2_full"].isin(same_images_path)
    ].reset_index(drop=True)
    print("Duplicate images removed.")
    return matches


####################################################################################################
# For the Q5-add_results_to_db.py script


def rename_h5_keys(h5_file_path, key_mapping):
    """
    Renames groups in the HDF5 file based on the provided key mapping.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        key_mapping (dict): Dictionary mapping old keys to new keys.
    """
    if not os.path.exists(h5_file_path):
        logging.error(f"HDF5 file not found: {h5_file_path}")
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    try:
        # Open the file in append mode so that it can be modified
        with h5py.File(h5_file_path, "a") as hdf5_file:
            for old_key, new_key in key_mapping.items():
                if old_key in hdf5_file:
                    # Check if the new key already exists
                    if new_key in hdf5_file:
                        logging.warning(
                            f"New key '{new_key}' already exists in the HDF5 file. Skipping renaming for '{old_key}'."
                        )
                    else:
                        hdf5_file.move(old_key, new_key)
                        logging.info(f"Renamed key '{old_key}' to '{new_key}'.")
                else:
                    logging.warning(
                        f"Old key '{old_key}' not found in the HDF5 file. Skipping."
                    )
        logging.info("All possible keys have been renamed successfully.")
    except Exception as e:
        logging.error(f"Error while renaming keys in HDF5 file: {e}")
        raise


def add_new_wireframe(device, wireframe_path, new_wireframe, new_unique_image_ids):
    logging.info("Adding new precomputed features to HDF5...")
    if os.path.exists(wireframe_path):
        try:
            with h5py.File(wireframe_path, "a") as hdf5_file:
                for image_id in new_unique_image_ids:
                    if image_id in new_wireframe:
                        if image_id in hdf5_file:
                            continue
                            # del hdf5_file[image_id]
                        grp = hdf5_file.create_group(image_id)
                        grp.create_dataset(
                            "image_size",
                            data=new_wireframe[image_id]["image_size"],
                        )
                        # grp.create_dataset("image", data=new_wireframe[image_id]["image"].cpu().numpy())
                        grp.create_dataset(
                            "keypoints",
                            data=new_wireframe[image_id]["keypoints"].cpu().numpy(),
                        )
                        grp.create_dataset(
                            "keypoint_scores",
                            data=new_wireframe[image_id]["keypoint_scores"]
                            .cpu()
                            .numpy(),
                        )
                        grp.create_dataset(
                            "descriptors",
                            data=new_wireframe[image_id]["descriptors"].cpu().numpy(),
                        )
                        grp.create_dataset(
                            "lines",
                            data=new_wireframe[image_id]["lines"].cpu().numpy(),
                        )
                        grp.create_dataset(
                            "line_scores",
                            data=new_wireframe[image_id]["line_scores"].cpu().numpy(),
                        )
                        grp.create_dataset(
                            "pl_associativity",
                            data=new_wireframe[image_id]["pl_associativity"]
                            .cpu()
                            .numpy(),
                        )
                        grp.create_dataset(
                            "lines_junc_idx",
                            data=new_wireframe[image_id]["lines_junc_idx"]
                            .cpu()
                            .numpy(),
                        )
            logging.info("Added all new precomputed features to HDF5.")
        except Exception as e:
            logging.error(f"Error adding new precomputed features: {e}")
            raise
    else:
        logging.error(f"Precomputed features file not found: {wireframe_path}")
        raise FileNotFoundError(
            f"Precomputed features file not found: {wireframe_path}"
        )


def load_regression_model(path):
    import joblib

    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
