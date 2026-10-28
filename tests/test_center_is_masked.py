"""The check that decides whether SAM handed back the animal or its background.

Grounded-SAM inverts the mask when `center_is_masked` says it misses the
subject. These cases need no weights and no data:

    python tests/test_center_is_masked.py      # or: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.image_preparation_utils import center_is_masked  # noqa: E402

H, W = 1536, 2048   # a camera-trap frame (Reconyx)


def animal(x0, y0, x1, y1):
    """An elliptical 'animal' filling a box, and that box as GroundingDINO gives it."""
    yy, xx = np.mgrid[:H, :W]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    return mask, np.array([x0, y0, x1, y1], dtype=np.float32)


def test_off_centre_animal_keeps_its_mask():
    mask, box = animal(150, 900, 750, 1400)        # bottom left of the frame
    assert not mask[H // 2, W // 2]
    assert not center_is_masked(mask)              # image centre: would invert it
    assert center_is_masked(mask, box=box)         # box centre: keeps it


def test_background_mask_is_still_inverted():
    mask, box = animal(150, 900, 750, 1400)
    in_box = np.zeros_like(mask)
    in_box[900:1400, 150:750] = True
    background = in_box & ~mask                    # SAM picked the surroundings
    assert not center_is_masked(background, box=box)


def test_centred_animal_is_judged_as_before():
    mask, box = animal(624, 468, 1424, 1068)
    np.random.seed(0)
    before = center_is_masked(mask)
    np.random.seed(0)
    assert before and center_is_masked(mask, box=box) == before


def test_box_off_the_edge_of_the_frame():
    mask, box = animal(-60, 1250, 400, 1600)       # cut by the bottom-left corner
    assert center_is_masked(mask, box=box)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok ", name)
