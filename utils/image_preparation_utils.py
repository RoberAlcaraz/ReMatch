import os
import cv2
import numpy as np
import pandas as pd
import glob
import torch
import torchvision
import torch.nn.functional as F
import logging
import sys


from segment_anything import sam_model_registry
from .automatic_mask_and_probability_generator import (
    SamAutomaticMaskAndProbabilityGenerator,
)


import params.image_preparation_params as params

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Stream to stdout
    ],
    force=True,
)


# Grounded Segment Anything Model (GSAM) utility functions
# Prompting SAM with detected boxes
def segment(sam_predictor, image, xyxy, invert=False):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        mask = masks[index]
        if invert:
            mask = np.logical_not(mask)
        result_masks.append(mask)
    return np.array(result_masks)


def center_is_masked(mask, patch_size=20, sample_points=10, threshold=0.3):
    H, W = mask.shape
    cx, cy = W // 2, H // 2
    half = patch_size // 2

    ys = np.random.randint(cy - half, cy + half, size=sample_points)
    xs = np.random.randint(cx - half, cx + half, size=sample_points)

    values = [mask[y, x] for y, x in zip(ys, xs)]
    ratio = np.mean(values)
    # print(ratio)
    return ratio > threshold


def GSAM_segmentation(grounding_dino_model, sam_predictor, classes):
    # Create directory
    os.makedirs(params.SEGMENTED_IMAGES_FOLDER, exist_ok=True)

    # Process each image
    possible_extensions = [".jpg", ".JPG", ".png"]
    for image_path in glob.glob(params.RAW_IMAGES_FOLDER + "/*/*", recursive=True):
        if not any(image_path.endswith(ext) for ext in possible_extensions):
            print(f"Skipping non-image file: {image_path}")
            continue
        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()

        # Get image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path).split("/")[-1]

        image = cv2.imread(image_path)

        gsam_image_path = (
            f"{params.SEGMENTED_IMAGES_FOLDER}/{image_dir}/{image_name}.png"
        )
        os.makedirs(os.path.dirname(gsam_image_path), exist_ok=True)

        # Skip processing if the image has already been processed
        if os.path.exists(gsam_image_path):
            # print(f"Image {image_name} already processed")
            continue
        print(f"Processing image: {image_dir}/{image_name}")

        # Detect objects using GroundingDINO
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=params.BOX_THRESHOLD,
            text_threshold=params.TEXT_THRESHOLD,
        )

        # Apply Non-Maximum Suppression (NMS) to filter detections
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                params.NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        # Update detections after NMS
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        try:
            # Select the detection with the highest confidence
            max_confidence_idx = np.argmax(detections.confidence)
            detections.xyxy = detections.xyxy[max_confidence_idx].reshape(1, 4)
            detections.confidence = detections.confidence[max_confidence_idx].reshape(1)
            detections.class_id = detections.class_id[max_confidence_idx].reshape(1)

            if detections.confidence[0] < 0.6:
                print(
                    f"Low confidence detection: {detections.confidence[0]} for {gsam_image_path}"
                )
                # continue
            # Convert detections to masks using SAM predictor
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )

            if not center_is_masked(detections.mask[0]):
                # Recompute inverted mask if center not covered
                detections.mask = segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy,
                    invert=True,
                )

            # Extract the first mask (assuming it's the one we want)
            mask = detections.mask[0]

            # Create an alpha channel based on the mask
            alpha_channel = (mask * 255).astype(np.uint8)

            # Add the alpha channel to the image
            transparent_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            transparent_image[:, :, 3] = alpha_channel

            # Crop the image to the bounding box
            x_min, y_min, x_max, y_max = detections.xyxy[0].astype(int)
            transparent_image = transparent_image[y_min:y_max, x_min:x_max]

            # Save the segmented image
            print(f"Saving segmented image: {gsam_image_path}")
            cv2.imwrite(gsam_image_path, transparent_image)

        except Exception as e:
            print(f"Error processing image: {image_name}, Error: {str(e)}")
            continue


############################################################################################
# YOLO utility functions
def scale_image_torch(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
      masks (torch.Tensor): resized and padded masks/images, [c, h, w].
      im0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
      masks (torch.Tensor): The masks that are being returned.
    """
    if len(masks.shape) < 3:
        raise ValueError(
            f'"len of masks shape" should be 3, but got {len(masks.shape)}'
        )
    im1_shape = masks.shape
    if im1_shape[1:] == im0_shape:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(
            im1_shape[1] / im0_shape[0], im1_shape[2] / im0_shape[1]
        )  # gain  = old / new
        pad = (im1_shape[2] - im0_shape[1] * gain) / 2, (
            im1_shape[1] - im0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[1] - pad[1]), int(im1_shape[2] - pad[0])

    masks = masks[:, top:bottom, left:right]
    if masks.shape[1:] != im0_shape:
        masks = F.interpolate(
            masks[None], im0_shape, mode="bilinear", align_corners=False
        )[0]

    return masks


def YOLO_segmentation(
    yolo_model_path, original_images_folder, segmented_images_folder, results_path
):
    # USE ROBOFLOW ENVIRONMENT!! (remember that we changed some parts in the code to avoid horizontal cuts)
    from ultralytics import YOLO

    # Force device selection explicitly. Ultralytics may keep weights on CPU unless asked.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(yolo_model_path)
    try:
        model.to(device)
    except Exception:
        # Older/newer ultralytics versions vary; we'll still pass device to predict below.
        pass
    logging.info("Model loaded")
    try:
        logging.info(
            f"YOLO device: {next(model.model.parameters()).device} (requested: {device})"
        )
    except Exception:
        logging.info(f"YOLO requested device: {device}")

    good_images = []

    images_with_errors = []
    error_types = []
    conf = []

    # Keep images in memory (avoid writing to disk for every image).
    resized_images = []
    segmented_image_paths = []

    # Process each lizard in the island
    for image_path in glob.glob(f"{original_images_folder}/*"):
        # Get the basename of the image
        image_basename = os.path.basename(image_path)
        print(f"Processing image: {image_basename}")
        # Get the folder name
        folder_name = os.path.basename(os.path.dirname(image_path))

        # Add the path of the resized image to the list
        segmented_image_name = f"{os.path.splitext(image_basename)[0]}.png"
        segmented_image_path = f"{segmented_images_folder}/{segmented_image_name}"

        if os.path.exists(segmented_image_path):
            logging.info(f"Processed image already exists: {segmented_image_path}")
            continue

        # Read the image
        image = cv2.imread(image_path)

        # Resize the image
        # resized_image = cv2.resize(image, (2048, 2048))
        resized_image = image

        # Add the image to the list (in-memory)
        resized_images.append(resized_image)
        segmented_image_paths.append(segmented_image_path)

    # Run YOLO model on the resized images
    if len(resized_images) == 0:
        logging.info("No images to process. Exiting...")
        return

    # Process images in small batches to avoid CUDA OOM on large sets.
    BATCH_SIZE = 8
    results = []
    for batch_start in range(0, len(resized_images), BATCH_SIZE):
        batch_imgs = resized_images[batch_start : batch_start + BATCH_SIZE]
        logging.info(
            f"YOLO batch {batch_start // BATCH_SIZE + 1}/"
            f"{(len(resized_images) + BATCH_SIZE - 1) // BATCH_SIZE}  "
            f"({len(batch_imgs)} images)"
        )
        batch_results = model(batch_imgs, device=device)
        results.extend(batch_results)
        torch.cuda.empty_cache()

    # Extract masks and crop them from the results
    for idx, result in enumerate(results):
        try:
            # if we have more than 1 detection, we skip the image
            if len(result.boxes.cls) > 1:
                print(
                    f"More than 1 detection found for image {segmented_image_paths[idx]}. Skipping..."
                )
                images_with_errors.append(segmented_image_paths[idx])
                error_types.append(">1detection")
                conf.append(max(result.boxes.conf.cpu().numpy()))
                # continue
            # if the probability is less than 0.9, we skip the image
            elif result.boxes.conf[0] < 0.9:
                print(
                    f"Low confidence detection found for image {segmented_image_paths[idx]}. Skipping..."
                )
                images_with_errors.append(segmented_image_paths[idx])
                error_types.append("low_confidence")
                conf.append(max(result.boxes.conf.cpu().numpy()))
                # continue

            # Get mask and original image from the result
            mask = result.masks.data  # Get the single mask
            orig_img = result.orig_img  # Get original image

            if len(mask) > 1:
                # Merge all masks into a single one
                merged_mask = np.zeros_like(mask[0].cpu().numpy())
                for i in range(mask.shape[0]):
                    merged_mask = np.maximum(merged_mask, mask[i].cpu().numpy())

                # Merged mask to tensor
                mask = torch.tensor([merged_mask], device=mask.device)

            mask_resized = scale_image_torch(
                mask,
                orig_img.shape[:2],
            )
            mask_resized = mask_resized[0].cpu().numpy()

            height, width, _ = orig_img.shape

            # Convert mask to binary format (0 and 255)
            binary_mask = (mask_resized > 0).astype(np.uint8) * 255

            # Find all connected components in the mask
            num_labels, labels_im = cv2.connectedComponents(binary_mask)

            # Keep track of the largest component (excluding the background)
            labels = list(range(1, num_labels))
            component_sizes = [np.sum(labels_im == label) for label in labels]

            # Get the components that are greater than the 0.25*max_size
            max_size = max(component_sizes)
            labels = np.array(labels)[component_sizes > 0.25 * max_size]
            component_sizes = np.array(component_sizes)[
                component_sizes > 0.25 * max_size
            ]

            # Create a mask for each component
            component_masks = [
                (labels_im == label).astype(np.uint8) * 255 for label in labels
            ]

            # Check which component is uppermost
            uppermost_component = np.argmin(
                [np.min(np.where(mask > 0)[0]) for mask in component_masks]
            )
            uppermost_component = component_masks[uppermost_component]

            # Find the non-zero coordinates in the largest component mask
            ys, xs = np.where(uppermost_component > 0)

            if len(xs) > 0 and len(ys) > 0:
                # Determine the bounding box of the largest component
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                # Crop the original image based on the mask region
                cropped_img = orig_img[y1:y2, x1:x2]

                # Crop the mask as well
                cropped_mask = uppermost_component[y1:y2, x1:x2]

                # Create an RGBA image where the mask is applied
                rgba_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
                rgba_image[:, :, 3] = (
                    cropped_mask  # Apply the mask as the alpha channel
                )

                # Save the RGBA image
                segmented_image_path = segmented_image_paths[idx]

                logging.info(f"Saving processed image to {segmented_image_path}")
                cv2.imwrite(
                    segmented_image_path,
                    rgba_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                good_images.append(segmented_image_path)
            else:
                logging.info(
                    f"No valid mask found for cropping: {segmented_image_path[idx]}"
                )
                images_with_errors.append(image_path)
                error_types.append("no_valid_mask")
                conf.append(0)

            # Clearing cache every image can slow things down; only do it if you see OOMs.

        except Exception as e:
            logging.info(
                f"Error processing lizard {segmented_image_path[idx]}: {e}. Continuing to the next image..."
            )
            images_with_errors.append(image_path)
            error_types.append("exception")
            conf.append(0)

    # No temporary images to remove.

    # Ensure results folder exists
    os.makedirs(results_path, exist_ok=True)

    # Save the list of images with errors
    with open(f"{results_path}/images_with_errors_YOLO.log", "w") as f:
        for image_path in images_with_errors:
            f.write(f"{image_path}\n")
            logging.info(f"Image with error: {image_path}")
        logging.info(f"Total images with errors: {len(images_with_errors)}")
        logging.info("List of images with errors saved to images_with_errors.log")

    # Save the list of error types as df
    df = pd.DataFrame({"image": images_with_errors, "error": error_types, "conf": conf})
    df.to_csv(f"{results_path}/images_with_errors_YOLO.csv", index=False)

    # Save the list of good images
    with open(f"{results_path}/good_images_YOLO.log", "w") as f:
        for image_path in good_images:
            f.write(f"{image_path}\n")
        print(f"Total good images: {len(good_images)}")
        print("List of good images saved to good_images.log")


############################################################################################
# Pattern extraction utility functions
def rotate_image(gray0, angle):
    (h, w) = gray0.shape[:2]
    center = (w // 2, h // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    gray0 = cv2.warpAffine(gray0, rotation_matrix, (w, h))
    return gray0


def calculate_rotation_angle(image_path):
    # Load the image
    # image_path = './IMG_5967.png'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Handle alpha channel if it exists
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image[np.where(alpha == 0)] = [0, 0, 0]
        image[np.where(alpha > 0)] = [255, 255, 255]

    # --- me quedo con la parte de arriba
    image_width = image.shape[1]
    image = image[0 : image_width // 4, 0:image_width]

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Initialize a dictionary to store the minimum y for each x
    min_y_per_x = {}

    # Find minimum y for each x where edges exist
    rows, cols = edges.shape
    for y in range(rows):  # Iterate over y (rows)
        for x in range(cols):  # Iterate over x (columns)
            if edges[y, x] > 0:  # Check if it is an edge pixel
                if x not in min_y_per_x:  # If x is not yet in the dictionary
                    min_y_per_x[x] = y
                else:
                    min_y_per_x[x] = min(min_y_per_x[x], y)

    # Create a new image to visualize the minimum y edges
    selected_edges = np.zeros_like(edges)

    # Draw the selected edges (min y for each x) in white
    for x, y in min_y_per_x.items():
        selected_edges[y, x] = 255

    # --- saco coordenadas de los puntos seleccionados y ordeno por x
    selected_edges = selected_edges.astype(np.uint8)
    coords = np.argwhere(selected_edges > 0)
    coords = coords[np.argsort(coords[:, 1])]

    # --- ahora sé que hay varias lineas dentro de las coords. Quiero separar estar coords por lineas con un threshold en la distancia en y
    # --- calculo la diferencia de y entre cada punto y el anterior
    dy = np.abs(np.diff(coords[:, 0]))

    # --- identifico los dy que son outliers y separo las lineas
    indi = np.where(dy > 50)[0]
    lines = np.split(coords, indi + 1)

    # --- me quedo con la linea que la diferencia entre su x sea mayor
    L = []
    for line in lines:
        L.append(line[-1, 1] - line[0, 1])

    # --- me quedo con la linea que la diferencia entre su x sea mayor
    max_line = np.argmax(L)
    lines = [lines[max_line]][0]

    # --- ahora quiero hacer un ajuste lineal a las coordenadas de la linea con trozos de 10 puntos
    numpun = 10
    M = []
    for i in range(0, lines.shape[0] - numpun):
        x = lines[i : i + numpun, 1]
        y = lines[i : i + numpun, 0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        M.append(m)

    # Remove the values that are far from the 95% percentile
    # M = M[(M > np.percentile(M, 5)) & (M < np.percentile(M, 95))]
    M = np.array(M)
    M = -np.median(M)

    angle = -np.arctan(M) * 180 / np.pi

    # Ensure the angle is within a reasonable range (e.g., 60 to 120 degrees)
    if 0 <= abs(angle) <= 45:
        print(f"Calculated Angle: {angle}")
        return angle
    else:
        print(
            "Detected angle out of expected range. Assuming image is already vertical."
        )
        print(f"Calculated Angle: {angle}")
        return 0


def extract_non_transparent_region(image):
    alpha = image[:, :, 3]
    # Get the bounding box of non-transparent pixels
    coords = cv2.findNonZero(alpha)
    x, y, w, h = cv2.boundingRect(coords)
    # Add some margin to the bounding box
    padding = 25
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    # Crop the image using the bounding box
    image = image[y : y + h, x : x + w]
    return image


def measure_contrast(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate the standard deviation of pixel intensities
    contrast = np.std(gray)
    return contrast


def apply_clahe(image, contrast_threshold=30.0):
    # Measure the contrast of the image
    contrast = measure_contrast(image)
    # print("Contrast: ", contrast)
    # Check if the contrast is below the threshold
    if contrast < contrast_threshold:
        # Apply CLAHE if contrast is low
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return rgb_clahe
    else:
        # Return the original image if contrast is sufficient
        return image


def normalize_image(image):
    # Normalize the image to the range [0, 1]
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)
    # Binarize the image
    image = np.where(image > 0.25, 0.0, 1.0).astype(np.float32)
    return image


def generate_binary_edges(edge_detection, filtered_masks):
    p_max = None
    for mask in filtered_masks:
        p = mask["prob"]
        if p_max is None:
            p_max = p
        else:
            p_max = np.maximum(p_max, p)

    # Generate edges for rotation
    edges = normalize_image(p_max)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    # Convert edges to a binary image for rotation
    edges = (edges * 255).astype(np.uint8)
    edges = cv2.bitwise_not(edges)
    edges = (edges > 0).astype(np.uint8)
    return edges


def extract_pattern_from_images(
    segmented_images_path,
    pattern_images_path,
    results_path,
    sam_checkpoint_path,
    edge_nms_path,
    device,
):
    import gc
    from scipy.spatial.distance import cdist
    from scipy.stats import zscore

    # Configure TensorFlow to avoid pre-allocating all GPU memory,
    # which would starve PyTorch (SAM) of VRAM.
    # By default ISR runs on CPU to prevent GPU contention with SAM.
    # Set ISR_USE_GPU=1 to allow TensorFlow to use GPU.
    import tensorflow as tf
    isr_use_gpu = os.environ.get("ISR_USE_GPU", "0") == "1"
    gpus = tf.config.list_physical_devices("GPU")
    if not isr_use_gpu and gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("ISR configured on CPU (set ISR_USE_GPU=1 to enable GPU).")
        except RuntimeError:
            pass
    else:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Memory growth must be set before GPUs are initialised

    from ISR.models import RDN

    sam = sam_model_registry["default"](checkpoint=sam_checkpoint_path)
    sam.to(device=device)

    # Initialize the edge detection model and SAM generator
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_nms_path)

    # Define the SamAutomaticMaskAndProbabilityGenerator
    generator = SamAutomaticMaskAndProbabilityGenerator(
        sam,
        pred_iou_thresh=0,
        pred_iou_thresh_filtering=False,
        stability_score_thresh=0.75,
        stability_score_thresh_filtering=True,
        min_mask_region_area=50,
        nms_threshold=0.3,
    )

    # Initialize the ISR model with unverified SSL context to prevent download errors
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    model = RDN(weights="noise-cancel")

    wrong_images = []
    good_images = []
    to_check_images = []
    mean_stability_scores = []
    mean_predicted_ious = []

    # Process each lizard in the island
    for image_path in glob.glob(f"{segmented_images_path}/*"):
        try: 
            # Get the basename of the image
            image_basename = os.path.basename(image_path)

            # Add the path of the resized image to the list
            scale_pattern_image_name = f"{os.path.splitext(image_basename)[0]}.png"
            scale_pattern_image_path = f"{pattern_images_path}/{scale_pattern_image_name}"

            if os.path.exists(scale_pattern_image_path):
                logging.info(f"Processed image already exists: {scale_pattern_image_path}")
                continue

            logging.info(f"Processing image {image_path}")
            rotation_angle = calculate_rotation_angle(image_path)

            # Load the image and apply preprocessing
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = rotate_image(image, rotation_angle)
            image_width = image.shape[1]

            if image.shape[2] == 4:
                image = extract_non_transparent_region(image)

                # Crop the image
                image = image[0 : (image_width // 2), :]

                # Extract again the non transparent region
                image = extract_non_transparent_region(image)

                # Remove the alpha channel
                alpha = image[:, :, 3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image[np.where(alpha == 0)] = [0, 0, 0]

            # Enhance the image using the ISR model
            image = model.predict(image)

            # Load the body mask (used later for filtering)
            body_mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            body_mask = rotate_image(body_mask, rotation_angle)
            if body_mask.shape[2] == 4:
                # Extract the non-transparent region
                body_mask = extract_non_transparent_region(body_mask)

                # Crop the image
                body_mask = body_mask[0 : (image_width // 2), :]

                # Extract again the non transparent region
                body_mask = extract_non_transparent_region(body_mask)

                # Remove the alpha channel
                alpha = body_mask[:, :, 3]
                body_mask = cv2.cvtColor(body_mask, cv2.COLOR_BGR2RGB)
                body_mask[np.where(alpha == 0)] = [0, 0, 0]
                body_mask[np.where(alpha > 0)] = [255, 255, 255]

            # Apply histogram equalization (CLAHE) to the colored image
            image = apply_clahe(image, contrast_threshold=20.0)

            # Generate masks using the SAM generator
            masks = generator.generate(image)

            # Filter out masks based on the body mask
            filtered_masks = []
            for mask_data in masks:
                mask = mask_data["segmentation"]

                # Ensure the mask is binary
                mask_binary = (mask > 0).astype(np.uint8)

                # Resize the body mask to match the size of the current mask
                resized_body_mask = cv2.resize(
                    body_mask,
                    (mask_binary.shape[1], mask_binary.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                if len(resized_body_mask.shape) == 3:
                    resized_body_mask = cv2.cvtColor(resized_body_mask, cv2.COLOR_RGB2GRAY)

                resized_body_mask = (resized_body_mask > 0).astype(np.uint8)

                # Ensure the mask is within the body mask
                intersection = cv2.bitwise_and(mask_binary, resized_body_mask)
                intersection_area = np.sum(intersection)
                original_mask_area = np.sum(mask_binary)

                if (
                    original_mask_area > 0
                    and (intersection_area / original_mask_area) > 0.9
                ):
                    filtered_masks.append(mask_data)

            # Remove masks with area less than the 20th percentile of area
            areas = [mask["area"] for mask in filtered_masks]
            if areas:
                filtered_masks = [
                    mask
                    for mask in filtered_masks
                    if mask["area"] >= np.quantile(areas, 0.2)
                ]

            # Create a probability map for the remaining masks
            edges = generate_binary_edges(edge_detection, filtered_masks)

            # Calculate the centroids of the filtered masks for outlier removal
            centroids = []
            for mask in filtered_masks:
                mask_coords = np.argwhere(mask["segmentation"] > 0)
                centroid = np.mean(mask_coords, axis=0)
                centroids.append(centroid)

            centroids = np.array(centroids)

            # Calculate the mean centroid
            mean_centroid = np.mean(centroids, axis=0)

            # Calculate distances from each centroid to the mean centroid
            distances = cdist(centroids, [mean_centroid], metric="euclidean").flatten()

            # Use z-score to identify outliers
            distance_z_scores = zscore(distances)
            threshold = 1.75  # Define the z-score threshold for outliers
            filtered_indices = np.where(distance_z_scores <= threshold)[0]

            # Keep only the masks that are not outliers
            filtered_masks = [filtered_masks[i] for i in filtered_indices]

            # Remove the largest mask
            if len(filtered_masks) > 1:
                filtered_masks = sorted(
                    filtered_masks, key=lambda x: x["area"], reverse=True
                )[1:]

            # Calculate the mean stability score and predicted IoU
            # to determine if the image is good enough
            mean_stability_score = np.mean(
                [mask["stability_score"] for mask in filtered_masks]
            )
            mean_predicted_iou = np.mean([mask["predicted_iou"] for mask in filtered_masks])
            if mean_stability_score < 0.94 and mean_predicted_iou < 0.9:
                logging.info(
                    f"Image {scale_pattern_image_path} with low stability and iou scores: {mean_stability_score}, {mean_predicted_iou}"
                )
                wrong_images.append(scale_pattern_image_path)
                # continue
            elif mean_stability_score < 0.95 or mean_predicted_iou < 0.95:
                logging.info(
                    f"Image {scale_pattern_image_path} to check: stability and iou scores: {mean_stability_score}, {mean_predicted_iou}"
                )
                to_check_images.append(scale_pattern_image_path)
                mean_stability_scores.append(mean_stability_score)
                mean_predicted_ious.append(mean_predicted_iou)
            else:
                good_images.append(scale_pattern_image_path)

            # Generate the final edges
            edges = generate_binary_edges(edge_detection, filtered_masks)

            # Get the min and max positions of the edges
            ymin, xmin = np.min(np.argwhere(edges > 0), axis=0)
            ymax, xmax = np.max(np.argwhere(edges > 0), axis=0)

            # Add padding to the bounding box
            padding = 25
            ymin = max(ymin - padding, 0)
            xmin = max(xmin - padding, 0)
            ymax = min(ymax + padding, edges.shape[0])
            xmax = min(xmax + padding, edges.shape[1])

            # Crop the image using the bounding box
            cropped_edges = edges[ymin:ymax, xmin:xmax]

            cropped_edges_width = cropped_edges.shape[1]
            cropped_edges = cropped_edges[0:cropped_edges_width, :]

            logging.info(f"Saving processed image to {scale_pattern_image_path}")
            cv2.imwrite(scale_pattern_image_path, (cropped_edges * 255).astype(np.uint8))
            torch.cuda.empty_cache()
        except Exception as e:
            logging.info(
                f"Error processing image {image_path}: {e}. Adding to wrong images list."
            )
            wrong_images.append(image_path)
            continue

    # Save the wrong images
    with open(f"{results_path}/wrong_images.log", "a") as f:
        for image_path in wrong_images:
            f.write(f"{image_path}\n")

    # Save the good images
    with open(f"{results_path}/good_images.log", "a") as f:
        for image_path in good_images:
            f.write(f"{image_path}\n")

    # Save the images to check
    with open(f"{results_path}/to_check_images.log", "a") as f:
        for image_path in to_check_images:
            f.write(f"{image_path}\n")

    # Clean up GPU memory to prevent OOM when called multiple times
    del sam, generator, model
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()
    gc.collect()
    


############################################################################################
# Utility functions to split the images and save the paths
def split_dataset(base_dir, train_dir, test_dir, p_unseen=0.3, p_train_seen=0.7):
    """
    Splits the dataset from base_dir into train and test sets with two types of individuals:
      1. Unseen individuals (all images go to test) chosen with probability p_unseen.
      2. Seen individuals (images are split randomly: each image goes to train with probability p_train_seen,
         and to test otherwise).

    Also collects relative paths (with respect to the train_dir) for training images.

    Parameters:
      base_dir (str): Path to the base directory containing individual folders.
      train_dir (str): Destination directory for training images.
      test_dir (str): Destination directory for test images.
      p_unseen (float): Fraction of individuals that will be completely held out (i.e., all images to test).
      p_train_seen (float): For individuals that appear in both sets, the probability an image is assigned to train.

    Returns:
      A list of relative paths (from train_dir) of the training images.
    """
    import random
    import shutil

    individuals = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    total_images = 0
    train_images = 0
    test_images = 0
    train_paths = []

    for indiv in individuals:
        indiv_path = os.path.join(base_dir, indiv)
        images = [f for f in os.listdir(indiv_path) if f.lower().endswith(".png")]
        num_images = len(images)
        total_images += num_images

        if random.random() < p_unseen:
            # Unseen individual: all images go to test
            for img in images:
                src_img = os.path.join(indiv_path, img)
                target_dir = os.path.join(test_dir, indiv)
                os.makedirs(target_dir, exist_ok=True)
                dest_img = os.path.join(target_dir, img)
                shutil.copy(src_img, dest_img)
                print(f"Copied (unseen) {src_img} to {dest_img}")
            test_images += num_images
        else:
            # Seen individual: each image is split between train and test
            for img in images:
                src_img = os.path.join(indiv_path, img)
                if random.random() < p_train_seen:
                    target_dir = os.path.join(train_dir, indiv)
                    train_images += 1
                    # Compute relative path with respect to train_dir
                    rel_path = os.path.relpath(
                        os.path.join(target_dir, img), start=train_dir
                    )
                    train_paths.append(rel_path)
                else:
                    target_dir = os.path.join(test_dir, indiv)
                    test_images += 1
                os.makedirs(target_dir, exist_ok=True)
                dest_img = os.path.join(target_dir, img)
                shutil.copy(src_img, dest_img)
                print(f"Copied {src_img} to {dest_img}")

    print("\nSplit summary:")
    print(f"Total images: {total_images}")
    print(f"Training images: {train_images} ({(train_images/total_images)*100:.2f}%)")
    print(f"Testing images: {test_images} ({(test_images/total_images)*100:.2f}%)")

    return train_paths


def group_test_images(test_dir, group_size=40):
    """
    Gathers all test images from subfolders of test_dir, shuffles them, and groups them
    into new subfolders (within test_dir) where each folder contains at most group_size images.

    Parameters:
      test_dir (str): Path to the test directory where images are stored in individual folders.
      group_size (int): Number of images per group folder.
    """
    import random
    import shutil

    # Collect all test images (searching recursively)
    image_paths = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(".png"):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print("No test images found to group.")
        return

    random.shuffle(image_paths)

    group_index = 1
    for i in range(0, len(image_paths), group_size):
        group_folder = os.path.join(test_dir, f"group_{group_index}")
        os.makedirs(group_folder, exist_ok=True)
        for img_path in image_paths[i : i + group_size]:
            dest_img = os.path.join(group_folder, os.path.basename(img_path))
            shutil.copy(img_path, dest_img)
        group_index += 1


def save_image_paths(train_folder, test_folder, good_images_folder):
    for folder_name in os.listdir(test_folder):
        folder_path = os.path.join(test_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("group_"):
            log_file_path = os.path.join(
                good_images_folder, f"{folder_name}_good_images.log"
            )

            # Collect file paths
            file_paths = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))

            # Save file paths to .log file
            with open(log_file_path, mode="w") as log_file:
                for path in file_paths:
                    log_file.write(f"{path}\n")

    print("File paths have been saved to the results folder.")

    # Repeat the process for train data
    # Iterate through group folders
    for folder_name in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, folder_name)
        # print(folder_path)
        if os.path.isdir(folder_path):
            log_file_path = os.path.join("results/good_images/train_good_images.log")

            # Collect file paths
            file_paths = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))

            # Save file paths to .log file
            with open(log_file_path, mode="a") as log_file:
                for path in file_paths:
                    log_file.write(f"{path}\n")


def save_metadata(train_folder, test_folder):
    from wildlife_tools.data import WildlifeDataset

    class Test(WildlifeDataset):
        def create_catalogue(self) -> pd.DataFrame:
            df = pd.DataFrame(
                {
                    "image_id": image_id,
                    "identity": identity,
                    "path": path,
                }
            )
            return df

    test_images = [x for x in glob.glob(f"{test_folder}/*/*") if not "group" in x]
    # Remove "data/" from the path
    test_path = [x.replace("data/", "") for x in test_images]
    test_image_id = [x.split("/")[-1].split(".")[0] for x in test_images]
    test_identity = [x.split("/")[-2] for x in test_images]

    train_images = [x for x in glob.glob(f"{train_folder}/*/*") if not "group" in x]
    # Remove "data/" from the path
    train_path = [x.replace("data/", "") for x in train_images]
    train_image_id = [x.split("/")[-1].split(".")[0] for x in train_images]
    train_identity = [x.split("/")[-2] for x in train_images]

    path = train_path + test_path
    image_id = train_image_id + test_image_id
    identity = train_identity + test_identity

    metadata = Test("data")
    metadata.to_csv("data/metadata.csv", index=False)
