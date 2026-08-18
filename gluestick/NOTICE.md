# Vendored code: GlueStick

The contents of this directory are a vendored copy of
[cvg/GlueStick](https://github.com/cvg/GlueStick), released by the Computer
Vision and Geometry Lab (ETH Zurich) under the MIT License. The upstream
license text is reproduced in [`LICENSE`](LICENSE) and applies to this
directory only; the rest of the ReMatch repository is covered by the
[top-level LICENSE](../LICENSE).

If you use this code, please cite the original work:

```bibtex
@inproceedings{pautrat_suarez_2023_gluestick,
  title     = {{GlueStick}: Robust Image Matching by Sticking Points and Lines Together},
  author    = {Pautrat, R{\'e}mi and Su{\'a}rez, Iago and Yu, Yifan and Pollefeys, Marc and Larsson, Viktor},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

`models/superpoint.py` is in turn an inference re-implementation of SuperPoint
(DeTone et al., CVPRW 2018), originally released at
[MagicLeapResearch/SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
The pretrained SuperPoint and GlueStick weights are **not** redistributed here;
they are downloaded at runtime from the upstream GlueStick repository and remain
subject to their original terms.

## Modifications made for ReMatch

The vendored copy is not byte-identical to upstream. The changes are:

- `models/two_view_pipeline_precomputed_wireframe.py` — **added** by the ReMatch
  authors. A variant of `two_view_pipeline.py` that consumes wireframes loaded
  from an on-disk HDF5 cache instead of recomputing them, so that each image is
  described once and reused across all of its pairs.
- `models/superpoint.py`, `models/wireframe.py` — extended with optional HDF5
  read/write (`save_path`, `image_id` arguments) supporting the same cache.
- Formatting only (black) in the remaining files.
