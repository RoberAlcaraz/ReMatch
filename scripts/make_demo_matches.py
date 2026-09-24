"""Regenerate the precomputed match tables the demo notebooks ship with.

Matching is the slow part of every demo: on a CPU the bundled lizard gallery
alone is 1 128 pairs, about a quarter of an hour. So each demo can load its
match table from `data/precomputed/` instead of recomputing it (`RUN_MATCHING =
False`, the default). This script is what produces those tables, with exactly
the calls the notebooks make. Run it again whenever the bundled images, the
wireframe configuration or the matcher change:

    python scripts/make_demo_matches.py            # all three datasets
    python scripts/make_demo_matches.py lizard     # or just one: lizard, zebra

The zebra tables need the Grounded-SAM weights and the GroundingDINO package,
since the zebras are matched on their segmented crops (see the README).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # params/ holds paths relative to the repository root

import utils.demo_utils as D  # noqa: E402

OUT = ROOT / "data" / "precomputed"


def lizard(pipeline, dev):
    pattern = ROOT / "data" / "images-pattern"
    query = ROOT / "data" / "new" / "lizard_demo-pattern"
    truth = json.loads((ROOT / "data" / "lizard_demo_ground_truth.json").read_text())

    # demo-1: every pair of the gallery.
    individuals = sorted(p.name for p in pattern.iterdir()
                         if p.is_dir() and p.name.startswith("lizard_"))
    labelled = [(i, p) for i in individuals for p in D.list_images(pattern / i)]
    D.all_pairs_table(pipeline, labelled, dev) \
        .to_parquet(OUT / "lizard_demo_matches.parquet", index=False)

    # demo-2: every query against every gallery image.
    gallery = [(i, p) for i in truth["gallery_individuals"]
               for p in D.list_images(pattern / i)]
    D.query_vs_gallery_table(pipeline, D.list_images(query), gallery, dev) \
        .to_parquet(OUT / "lizard_demo_query_matches.parquet", index=False)


def zebra(pipeline, dev):
    gallery_dir = ROOT / "data" / "images"
    query_dir = ROOT / "data" / "new" / "zebra_demo"
    gal_src = ROOT / "data" / "images-segmented"
    qry_src = query_dir.parent / f"{query_dir.name}-segmented"

    # demo-3, stage 0: segment exactly as the notebook does.
    individuals = sorted(p.name for p in gallery_dir.iterdir() if p.is_dir())
    gsam = D.build_grounded_sam(dev)
    for i in individuals:
        D.segment_images(gsam, D.list_images(gallery_dir / i), gal_src / i, ["zebra"], dev)
    D.segment_images(gsam, D.list_images(query_dir), qry_src, ["zebra"], dev)
    del gsam
    D.empty_device_cache(dev)

    labelled = [(i, p) for i in individuals for p in D.list_images(gal_src / i)]
    D.all_pairs_table(pipeline, labelled, dev) \
        .to_parquet(OUT / "zebra_demo_matches.parquet", index=False)
    D.query_vs_gallery_table(pipeline, D.list_images(qry_src), labelled, dev) \
        .to_parquet(OUT / "zebra_demo_query_matches.parquet", index=False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["lizard", "zebra"]
    OUT.mkdir(parents=True, exist_ok=True)
    dev = D.device()
    for name in which:
        D.set_seed(0)
        D.ENHANCE_CONTRAST = True   # what every demo runs with
        print(f"--- {name}")
        {"lizard": lizard, "zebra": zebra}[name](D.build_pipeline(dev), dev)
    print(f"written to {OUT}")
