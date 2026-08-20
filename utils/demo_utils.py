"""Helpers shared by the two demo notebooks.

These wrap the same primitives the pipeline scripts use, with the on-disk
bookkeeping (the HDF5 wireframe cache and the LMDB match store) stripped out so
the notebooks stay readable. The demo keeps the wireframes in memory instead,
which the bundled dataset is small enough for (~3 MB per image). For a real
database use `scripts/`, which spills both caches to disk and can resume.

Everything else — the wireframe configuration, the 0.2 match filter, the four
features and their ordering — is exactly what the pipeline uses, and is read from
`params/`.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from gluestick import batch_to_np, numpy_image_to_torch
from gluestick.models.two_view_pipeline import TwoViewPipeline
from utils.utils import (  # noqa: F401
    read_transparent_img,
    resize_image,
    set_seed,
    enhance_contrast,
)

# The four features every ReMatch decision is built from, in the order the
# trained models expect them.
FEATURES = [
    "num_nonzero_points",
    "mean_point_prob",
    "num_nonzero_lines",
    "mean_line_prob",
]

# GlueStick's recommended default, adopted unchanged (Pautrat et al., 2023).
MATCH_PROB_THRESHOLD = 0.2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix in IMAGE_EXTENSIONS)


def build_pipeline(dev=None) -> TwoViewPipeline:
    """SuperPoint+LSD wireframe extraction and GlueStick matching in one module.

    Downloads the GlueStick and SuperPoint weights on first call.
    """
    from params.params import WIREFRAME_CONF, GLUESTICK_CONF

    conf = {
        "name": "two_view_pipeline",
        "use_lines": True,
        "extractor": {"name": "wireframe", **WIREFRAME_CONF},
        "matcher": GLUESTICK_CONF["matcher"],
        "ground_truth": {"from_pose_depth": False},
    }
    return TwoViewPipeline(conf).to(dev or device()).eval()


# CLAHE (contrast-limited adaptive histogram equalisation) applied to every
# image before its wireframe is extracted. On by default because that is what
# the published method does: `utils.compute_wireframe` always applies it, and so
# did the code behind the paper, for every species. Turn it off to see what it
# buys you - it lifts faint patterns out of flat, evenly-lit images and
# amplifies sensor noise in dark ones, and on the bundled lizard crops it alters
# about 96% of the pixels either way.
ENHANCE_CONTRAST = True


def load_gray(path, height=None, enhance=None):
    """Segmented image -> grayscale at the pipeline's working resolution.

    `enhance` overrides the module-level `ENHANCE_CONTRAST` for one call.
    """
    from params.params import IMAGE_HEIGHT_RESIZE

    gray = read_transparent_img(str(path))
    gray = resize_image(gray, height=height or IMAGE_HEIGHT_RESIZE)
    if ENHANCE_CONTRAST if enhance is None else enhance:
        gray = enhance_contrast(gray)   # same order as utils.compute_wireframe
    return gray


def match_pair(pipeline, path0, path1, dev=None):
    """Run the two-view pipeline on one pair. Returns (prediction, img0, img1)."""
    dev = dev or device()
    g0, g1 = load_gray(path0), load_gray(path1)
    x = {
        "image0": numpy_image_to_torch(g0).to(dev)[None],
        "image1": numpy_image_to_torch(g1).to(dev)[None],
    }
    with torch.no_grad():
        pred = pipeline(x)
    return batch_to_np(pred), g0, g1


def pair_features(pred) -> dict:
    """Reduce one pair's matches to the four features.

    Matches scoring below MATCH_PROB_THRESHOLD are discarded first, so the counts
    mean 'confident matches' and the means mean 'how confident they were'.
    """
    pts = np.asarray(pred["match_scores0"])
    lns = np.asarray(pred["line_match_scores0"])
    kp = pts[pts > MATCH_PROB_THRESHOLD]
    kl = lns[lns > MATCH_PROB_THRESHOLD]
    return {
        "num_nonzero_points": int(kp.size),
        "mean_point_prob": float(kp.mean()) if kp.size else 0.0,
        "num_nonzero_lines": int(kl.size),
        "mean_line_prob": float(kl.mean()) if kl.size else 0.0,
    }


def empty_device_cache(dev=None) -> None:
    """Return the allocator's cached blocks to the driver.

    CUDA and MPS keep freed blocks in a per-process cache. Because each image
    yields a different number of wireframe junctions, every pair allocates
    slightly differently sized tensors, so that cache grows and fragments over
    a long matching run even though live memory stays flat. The pipeline
    scripts call this after every image; the notebooks need it too.
    """
    dev = str(dev or device())
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def precompute_wireframes(pipeline, paths, dev=None, progress=True) -> dict:
    """Extract each image's wireframe once, keyed by path.

    The wireframe (SuperPoint keypoints + LSD lines merged into junctions)
    depends on one image only, so computing it inside the pair loop repeats the
    same work N-1 times per image. Costs about 3 MB per image to keep.
    """
    dev = dev or device()
    paths = list(dict.fromkeys(str(p) for p in paths))   # de-duplicate, keep order
    cache = {}
    for n, p in enumerate(paths, 1):
        if progress and (n % 10 == 0 or n == len(paths)):
            print(f"\r  wireframes {n}/{len(paths)}", end="", flush=True)
        gray = load_gray(p)
        with torch.no_grad():
            wf = pipeline.extractor({"image": numpy_image_to_torch(gray).to(dev)[None]})
        # valid_lines is produced by the extractor but not consumed by the matcher.
        cache[str(p)] = {k: v for k, v in wf.items() if k != "valid_lines"}
        del gray, wf
    if progress:
        print()
    empty_device_cache(dev)
    return cache


def match_cached(pipeline, wf0, wf1):
    """Match two precomputed wireframes. Returns the same dict as `match_pair`."""
    data = {"image_size0": wf0["image_size"], "image_size1": wf1["image_size"]}
    pred = {
        **{k + "0": v for k, v in wf0.items()},
        **{k + "1": v for k, v in wf1.items()},
    }
    with torch.no_grad():
        out = pipeline.matcher({**data, **pred})
    return batch_to_np(out)


def all_pairs_table(pipeline, labelled_images, dev=None, progress=True) -> pd.DataFrame:
    """Every unordered pair of a labelled gallery -> a feature table.

    Wireframes are extracted once per image and reused across every pair, the
    way `scripts/P2-pattern_matching.py` does it with its HDF5 cache. Only the
    matching itself is quadratic: a gallery of N images costs N wireframes and
    N(N-1)/2 matches.
    """
    dev = dev or device()
    combos = list(itertools.combinations(labelled_images, 2))
    cache = precompute_wireframes(pipeline, [p for _, p in labelled_images], dev, progress)

    # One update per ~1% of the run: a fixed interval would emit thousands of
    # output messages on a real gallery, which is itself enough to sink a
    # notebook front-end.
    step = max(25, len(combos) // 100)
    rows = []
    for n, ((id_a, pa), (id_b, pb)) in enumerate(combos, 1):
        if progress and (n % step == 0 or n == len(combos)):
            print(f"\r  matched {n}/{len(combos)} pairs", end="", flush=True)
        pred = match_cached(pipeline, cache[str(pa)], cache[str(pb)])
        rows.append({
            "img1_full": f"{id_a}/{Path(pa).name}",
            "img2_full": f"{id_b}/{Path(pb).name}",
            "id1": id_a,
            "id2": id_b,
            "same": id_a == id_b,
            **pair_features(pred),
        })
        del pred
        if n % 200 == 0:
            empty_device_cache(dev)
    if progress:
        print()
    del cache
    empty_device_cache(dev)
    return pd.DataFrame(rows)


def query_vs_gallery_table(pipeline, query_images, gallery_images, dev=None, progress=True):
    """Match each query image against every gallery image.

    `query_images` is a list of paths, `gallery_images` a list of (identity, path).
    Wireframes are extracted once per image, as in `all_pairs_table`.
    """
    dev = dev or device()
    cache = precompute_wireframes(
        pipeline, list(query_images) + [g for _, g in gallery_images], dev, progress
    )

    rows = []
    total = len(query_images) * len(gallery_images)
    step = max(25, total // 100)
    n = 0
    for q in query_images:
        for gid, g in gallery_images:
            n += 1
            if progress and (n % step == 0 or n == total):
                print(f"\r  matched {n}/{total} query-gallery pairs", end="", flush=True)
            pred = match_cached(pipeline, cache[str(q)], cache[str(g)])
            rows.append({
                "query": Path(q).name,
                "gallery_image": f"{gid}/{Path(g).name}",
                "id2": gid,
                **pair_features(pred),
            })
            del pred
            if n % 200 == 0:
                empty_device_cache(dev)
    if progress:
        print()
    del cache
    empty_device_cache(dev)
    return pd.DataFrame(rows)


def draw_matches(pred, g0, g1, title=""):
    """Side-by-side match visualisation: points, then lines."""
    import matplotlib.pyplot as plt
    from gluestick.drawing import plot_images, plot_matches, plot_color_line_matches

    kp0, kp1, m0 = pred["keypoints0"], pred["keypoints1"], pred["matches0"]
    valid = m0 != -1
    mk0, mk1 = kp0[valid], kp1[m0[valid]]

    l0, l1, lm = pred["lines0"], pred["lines1"], pred["line_matches0"]
    lvalid = lm != -1
    ml0, ml1 = l0[lvalid], l1[lm[lvalid]]

    plot_images([g0, g1], [f"{title}", f"{len(mk0)} matched points"], dpi=110, pad=2.0)
    plot_matches(mk0, mk1, "lime", lw=0.4, ps=0)
    plt.show()

    plot_images([g0, g1], ["", f"{len(ml0)} matched lines"], dpi=110, pad=2.0)
    plot_color_line_matches([ml0, ml1], lw=2)
    plt.show()


# ---------------------------------------------------------------------------
# Grounded-SAM segmentation, for demo-3 (a species with no pattern extraction).
#
# GroundingDINO is not on PyPI and most users will never install it, so both
# imports below are deferred: demo-1 and demo-2 import this module happily
# without either package present.
# ---------------------------------------------------------------------------

def build_grounded_sam(dev=None, dino_config=None, dino_weights=None,
                       sam_weights=None, sam_version="vit_h"):
    """Load GroundingDINO and SAM once, ready for `segment_images`.

    Paths default to the ones in `params/image_preparation_params.py`, so this
    picks up the weights the README tells you to drop into `models/`.
    """
    import params.image_preparation_params as P
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor

    dev = dev or device()

    # Neither model downloads its own weights, unlike GlueStick and SuperPoint,
    # so say which file is missing and where it comes from rather than letting
    # a bare FileNotFoundError surface from inside the loader.
    needed = {
        dino_config or P.GROUNDING_DINO_CONFIG_PATH:
            "ships with this repository - restore it from git",
        dino_weights or P.GROUNDING_DINO_CHECKPOINT_PATH:
            "https://github.com/IDEA-Research/GroundingDINO#luggage-checkpoints",
        sam_weights or P.SAM_CHECKPOINT_PATH:
            "https://github.com/facebookresearch/segment-anything#model-checkpoints",
    }
    absent = [f"  {p}  <-  {where}" for p, where in needed.items() if not Path(p).exists()]
    if absent:
        raise FileNotFoundError(
            "Grounded-SAM needs weights that are not in the repository:\n"
            + "\n".join(absent)
            + "\n\nIf your images are already segmented or cropped, you do not need "
              "any of this:\nset SEGMENT = False and the notebook will use them as "
              "they are."
        )

    dino = Model(
        model_config_path=dino_config or P.GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=dino_weights or P.GROUNDING_DINO_CHECKPOINT_PATH,
        device=str(dev),
    )
    sam = sam_model_registry[sam_version](checkpoint=sam_weights or P.SAM_CHECKPOINT_PATH)
    sam.to(device=dev)
    return dino, SamPredictor(sam)


def segment_images(models, paths, out_dir, classes, dev=None, progress=True,
                   box_threshold=None, text_threshold=None, nms_threshold=None):
    """Text-prompted segmentation: raw photographs -> RGBA crops of the animal.

    Same procedure as `utils.image_preparation_utils.GSAM_segmentation`, which
    `scripts/P1` uses — GroundingDINO proposes boxes for `classes`, NMS keeps
    the best, SAM turns it into a mask, and the image is cropped to the box with
    the mask as its alpha channel. Written per-file rather than per-folder so a
    notebook can report progress and segment a query batch separately.

    Returns (written_paths, skipped) where `skipped` lists (path, reason).
    """
    import numpy as _np
    import torch as _torch
    import torchvision
    import params.image_preparation_params as P
    from utils.image_preparation_utils import segment, center_is_masked

    dino, sam_predictor = models
    dev = dev or device()
    box_th = P.BOX_THRESHOLD if box_threshold is None else box_threshold
    txt_th = P.TEXT_THRESHOLD if text_threshold is None else text_threshold
    nms_th = P.NMS_THRESHOLD if nms_threshold is None else nms_threshold

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    for n, p in enumerate(map(Path, paths), 1):
        if progress:
            print(f"\r  segmenting {n}/{len(paths)}  {p.name:<28}", end="", flush=True)
        dst = out_dir / f"{p.stem}.png"
        if dst.exists():                      # resumable, like the pipeline
            written.append(dst)
            continue

        image = cv2.imread(str(p))
        if image is None:
            skipped.append((p, "unreadable")); continue

        det = dino.predict_with_classes(image=image, classes=list(classes),
                                        box_threshold=box_th, text_threshold=txt_th)
        if det.xyxy is None or len(det.xyxy) == 0:
            skipped.append((p, f"no '{'/'.join(classes)}' detected")); continue

        keep = torchvision.ops.nms(_torch.from_numpy(det.xyxy),
                                   _torch.from_numpy(det.confidence),
                                   nms_th).numpy().tolist()
        det.xyxy, det.confidence = det.xyxy[keep], det.confidence[keep]

        best = int(_np.argmax(det.confidence))
        box, conf = det.xyxy[best].reshape(1, 4), float(det.confidence[best])

        mask = segment(sam_predictor=sam_predictor,
                       image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=box)
        if not center_is_masked(mask[0]):
            # SAM occasionally returns the background; the pipeline retries inverted.
            mask = segment(sam_predictor=sam_predictor,
                           image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                           xyxy=box, invert=True)

        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (mask[0] * 255).astype(_np.uint8)
        x0, y0, x1, y1 = box[0].astype(int)
        crop = rgba[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            skipped.append((p, "empty crop")); continue

        cv2.imwrite(str(dst), crop)
        written.append(dst)
        if conf < P.MIN_DETECTION_CONFIDENCE:
            skipped.append((p, f"low confidence {conf:.2f} — review {dst.name}"))
        empty_device_cache(dev)

    if progress:
        print()
    return written, skipped


# ---------------------------------------------------------------------------
# Near-duplicate photographs.
# ---------------------------------------------------------------------------

from params.params import DUPLICATE_POINT_THRESHOLD  # noqa: E402  (single source)


def duplicate_pairs(matches, threshold=DUPLICATE_POINT_THRESHOLD):
    """Pairs matched so strongly that they are almost certainly one photograph.

    Two different photographs of the same animal share a few hundred keypoints
    at best. Past `threshold` the likeliest explanation is that the same image
    is in the gallery twice — a burst frame, a re-import, the same file under
    two names. Left in, it inflates the same-individual class with a comparison
    the model will never face in the field.
    """
    return matches[matches["num_nonzero_points"] > threshold].copy()


def drop_duplicate_images(matches, threshold=DUPLICATE_POINT_THRESHOLD):
    """Remove near-duplicates, keeping one copy of each.

    Same rule as `utils.utils.remove_duplicate_images`, which
    `scripts/P3-feature_aggregation.py` applies: drop the offending pairs, then
    drop every remaining row that touches the *first* image of such a pair. The
    asymmetry is deliberate — of two copies, one is removed and one survives to
    represent the animal.

    Returns (clean_matches, dropped_image_names, duplicate_pairs).
    """
    dups = duplicate_pairs(matches, threshold)
    if dups.empty:
        return matches.reset_index(drop=True), [], dups

    dropped = list(pd.unique(dups["img1_full"]))
    clean = matches.loc[~matches.index.isin(dups.index)]
    clean = clean.loc[
        ~clean["img1_full"].isin(dropped) & ~clean["img2_full"].isin(dropped)
    ]
    return clean.reset_index(drop=True), dropped, dups


def show_duplicate_pairs(dups, source_dir, max_pairs=6,
                         threshold=DUPLICATE_POINT_THRESHOLD):
    """Show each flagged pair side by side, so you can judge the call yourself.

    The threshold is a heuristic. If these look like genuinely different
    photographs, raise `threshold`; if obvious duplicates are getting through,
    lower it.
    """
    import matplotlib.pyplot as plt

    if dups.empty:
        print("No near-duplicate pairs found.")
        return

    source_dir = Path(source_dir)
    show = dups.sort_values("num_nonzero_points", ascending=False).head(max_pairs)
    fig, axes = plt.subplots(len(show), 2, figsize=(8, 3.4 * len(show)),
                             squeeze=False)
    for row, (_, r) in enumerate(zip(range(len(show)), show.iterrows())):
        _, r = r
        for col, (key, fate) in enumerate([("img1_full", "REMOVED"),
                                           ("img2_full", "kept")]):
            ax = axes[row][col]
            p = source_dir / r[key]
            if p.exists():
                ax.imshow(load_gray(p), cmap="gray")
            else:
                ax.text(.5, .5, f"not found:\n{r[key]}", ha="center", va="center",
                        fontsize=8)
            ax.set_title(f"{r[key]}\n{fate}", fontsize=8,
                         color="#a8412a" if fate == "REMOVED" else "#0d6e66")
            ax.axis("off")
        axes[row][0].set_ylabel(f"{r['num_nonzero_points']:.0f} pts")
    plt.suptitle(
        f"Pairs above {threshold} matched points — the left image is removed",
        y=1.005,
    )
    plt.tight_layout()
    plt.show()
