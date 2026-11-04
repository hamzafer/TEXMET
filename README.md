# TEXMET: Curated Textile Dataset from the Metropolitan Museum of Art

[![arXiv](https://img.shields.io/badge/arXiv-2611.04734-b31b1b.svg)](https://arxiv.org/abs/2611.04734) [![DOI](https://img.shields.io/badge/DOI-10.1145%2F3846195-blue.svg)](https://doi.org/10.1145/3846195) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-hzafar%2FTEXMET-yellow.svg)](https://huggingface.co/datasets/hzafar/TEXMET)

**Paper:** M. H. Zafar, D. Gigilashvili, J. Y. Hardeberg. *TEXMET: A Deep Learning Benchmark for Archaeological Textile Reconstruction.* ACM Journal on Computing and Cultural Heritage, 2026. [DOI 10.1145/3846195](https://doi.org/10.1145/3846195) · open-access preprint [arXiv:2611.04734](https://arxiv.org/abs/2611.04734)

**TEXMET** is a high-quality, manually curated dataset of textile and tapestry objects from the Metropolitan Museum of Art's Open Access collection. This dataset has been carefully cleaned, validated, and optimized for computer vision and deep learning applications.

## Quick Start: Download the Dataset

To download the TEXMET dataset, follow these steps:

**1. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the Download Script:**

```bash
python download.py
```

The script will download all 18,644 high-resolution images (approx. 26.5 GB) into a `TEXMET_DATASET` folder. Please ensure you have enough disk space.

For more options, run:
```bash
python download.py --help
```

## Dataset Overview

- **Total Images**: 18,644 high-resolution images
- **Unique Objects**: 18,644 Met objects, one primary image each (train 14,915 / validation 1,864 / test 1,865)
- **Average Resolution**: 3.3 MP (1557 x 1631 pixels)
- **Format**: 100% JPEG
- **Total Size**: 26.5 GB

The dataset metadata, including statistics and distributions, is available in `metadata.json`.

## Dataset Access
🤗 The full curated dataset (all 18,644 images with metadata and the paper's train / validation / test splits) is available on Hugging Face.

**[Open TEXMET on Hugging Face Datasets](https://huggingface.co/datasets/hzafar/TEXMET)**

## Citation and Credits

### Citing TEXMET
If you use TEXMET, please cite the paper:

```bibtex
@article{zafar2026texmet,
  author  = {Zafar, Muhammad Hamza and Gigilashvili, Davit and Hardeberg, Jon Yngve},
  title   = {{TEXMET}: A Deep Learning Benchmark for Archaeological Textile Reconstruction},
  journal = {ACM Journal on Computing and Cultural Heritage},
  year    = {2026},
  doi     = {10.1145/3846195},
  eprint  = {2611.04734},
  archivePrefix = {arXiv}
}
```

### Original Data Source
This dataset is a curated subset of textiles and tapestries from The Metropolitan Museum of Art's Open Access collection.

```bibtex
@misc{MET_OpenAccess_2024,
  author       = {{The Metropolitan Museum of Art}},
  title        = {Open Access Artworks Dataset},
  year         = {2024},
  howpublished = {\url{https://github.com/metmuseum/openaccess}},
  note         = {CC0 public-domain images and metadata; accessed July 2025},
}
```

## Legal & Licensing
- **License**: CC0 (Public Domain) - same as original MET data.
- **Usage Rights**: Free for any purpose without restriction.
- **Attribution**: The Metropolitan Museum of Art (recommended).

---
*Last Updated: November 4, 2026*