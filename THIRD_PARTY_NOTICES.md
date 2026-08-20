# Third-party notices

ReMatch itself is released under the MIT License (see [`LICENSE`](LICENSE)).
That license does **not** cover the third-party code vendored into this
repository, nor the model weights downloaded at runtime. This file records the
origin and terms of each of those components.

---

## Vendored source code

### `gluestick/` — GlueStick

| | |
|---|---|
| Upstream | https://github.com/cvg/GlueStick |
| Copyright | (c) 2023 Computer Vision and Geometry Lab, ETH Zurich |
| License | MIT — full text in [`gluestick/LICENSE`](gluestick/LICENSE) |
| Modified | Yes — see [`gluestick/NOTICE.md`](gluestick/NOTICE.md) |

Cite: Pautrat, Suárez, Yu, Pollefeys & Larsson, *GlueStick: Robust Image
Matching by Sticking Points and Lines Together*, ICCV 2023.

`gluestick/models/superpoint.py` is an inference re-implementation of SuperPoint
(DeTone, Malisiewicz & Rabinovich, CVPRW 2018), originally released at
[MagicLeapResearch/SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).

### `utils/automatic_mask_and_probability_generator.py`

| | |
|---|---|
| Original | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (`automatic_mask_generator.py`) |
| Copyright | (c) Meta Platforms, Inc. and affiliates |
| License | Apache License 2.0 — full text in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| Modified | Yes — extended to return per-mask probability maps used by the pattern-extraction step |

---

## Bundled data

### `data/images/zebra_*/` and `data/new/zebra_demo/` — GZGC plains zebras

| | |
|---|---|
| Upstream | https://lila.science/datasets/great-zebra-giraffe-id |
| Source | Great Zebra and Giraffe Count and ID (GZGC), Wild Me |
| License | [Community Data License Agreement – Permissive 1.0](https://cdla.dev/permissive-1-0/) — redistribution permitted |
| Redistributed | Yes — 43 photographs of 11 individuals, a subset for `demo-3-new-species.ipynb` |

Cite: Parham J, Crall J, Stewart C, Berger-Wolf T, Rubenstein DI. *Animal
population censusing at scale with citizen science and photographic
identification.* AAAI Spring Symposium — Technical Report, 2017.

The lizard images in `data/images-pattern/lizard_*/` and
`data/new/lizard_demo-pattern/` come from BalearicLizard, released by the
authors of this repository; see [`data/README.md`](data/README.md).

---

## Model weights (downloaded at runtime, not redistributed)

| Weights | Source | License |
|---|---|---|
| `checkpoint_GlueStick_MD.tar` | [cvg/GlueStick releases](https://github.com/cvg/GlueStick/releases) | MIT |
| `superpoint_v1.pth` | [cvg/GlueStick](https://github.com/cvg/GlueStick/tree/main/resources/weights) | Derived from SuperPoint; see upstream terms |
| `sam_vit_h_4b8939.pth` | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) | Apache-2.0 |
| `model.yml.gz` (structured edges) | [opencv/opencv_extra](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz) | BSD-3-Clause (OpenCV) |
| `yolo-segmentation.pt` | [huggingface.co/roberalcaraz/lizard-body-segmentation](https://huggingface.co/roberalcaraz/lizard-body-segmentation) | MIT (released by the ReMatch authors) |

> **AGPL notice.** ROI segmentation uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
> which is distributed under **AGPL-3.0**. Ultralytics is a runtime dependency
> (installed from PyPI, not vendored here), so it does not change the license of
> this repository, but AGPL-3.0 obligations apply to *your* deployment if you
> redistribute a service built on it. Ultralytics also sells a commercial
> license for that case.

---

## Bundled example data

`data/images-pattern/lizard_*` and `data/new/lizard_demo-pattern` contain 68
scale-pattern crops from **BalearicLizard**, released by the authors of this
repository and documented in *Scientific Data*
([doi:10.1038/s41597-026-07411-z](https://doi.org/10.1038/s41597-026-07411-z));
the full dataset is on
[Kaggle](https://www.kaggle.com/datasets/roberalcaraz/baleariclizard). See
[`data/README.md`](data/README.md).

The other four datasets the paper evaluates on are third-party and are **not**
redistributed here; their sources are in the paper's data availability statement.
