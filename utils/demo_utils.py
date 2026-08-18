"""Helpers shared by the two demo notebooks.

These wrap the same primitives the pipeline scripts use, with the on-disk
bookkeeping (the HDF5 wireframe cache and the LMDB match store) stripped out so
the notebooks stay readable. For a real database use `scripts/`, which caches
everything and can resume; the demo keeps wireframes in memory because the
bundled dataset is small.

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
from utils.utils import read_transparent_img, resize_image

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


def load_gray(path, height=None):
    """Segmented image -> grayscale at the pipeline's working resolution."""
    from params.params import IMAGE_HEIGHT_RESIZE

    gray = read_transparent_img(str(path))
    return resize_image(gray, height=height or IMAGE_HEIGHT_RESIZE)


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


def all_pairs_table(pipeline, labelled_images, dev=None, progress=True) -> pd.DataFrame:
    """Every unordered pair of a labelled gallery -> a feature table.

    `labelled_images` is a list of (identity, path). This is the quadratic step:
    a gallery of N images costs N(N-1)/2 matches.
    """
    combos = list(itertools.combinations(labelled_images, 2))
    rows = []
    for n, ((id_a, pa), (id_b, pb)) in enumerate(combos, 1):
        if progress and (n % 25 == 0 or n == len(combos)):
            print(f"\r  matched {n}/{len(combos)} pairs", end="", flush=True)
        pred, _, _ = match_pair(pipeline, pa, pb, dev)
        rows.append({
            "img1_full": f"{id_a}/{Path(pa).name}",
            "img2_full": f"{id_b}/{Path(pb).name}",
            "id1": id_a,
            "id2": id_b,
            "same": id_a == id_b,
            **pair_features(pred),
        })
    if progress:
        print()
    return pd.DataFrame(rows)


def query_vs_gallery_table(pipeline, query_images, gallery_images, dev=None, progress=True):
    """Match each query image against every gallery image.

    `query_images` is a list of paths, `gallery_images` a list of (identity, path).
    """
    rows = []
    total = len(query_images) * len(gallery_images)
    n = 0
    for q in query_images:
        for gid, g in gallery_images:
            n += 1
            if progress and (n % 25 == 0 or n == total):
                print(f"\r  matched {n}/{total} query-gallery pairs", end="", flush=True)
            pred, _, _ = match_pair(pipeline, q, g, dev)
            rows.append({
                "query": Path(q).name,
                "gallery_image": f"{gid}/{Path(g).name}",
                "id2": gid,
                **pair_features(pred),
            })
    if progress:
        print()
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
