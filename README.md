# TEXMET: Curated Textile Dataset from the Metropolitan Museum of Art

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
- **Unique Objects**: 1,697 textile/tapestry objects
- **Average Resolution**: 3.3 MP (1557 x 1631 pixels)
- **Format**: 100% JPEG
- **Total Size**: 26.5 GB

The dataset metadata, including statistics and distributions, is available in `metadata.json`.

## Dataset Access
🤗 A sample of the curated dataset is also available on Hugging Face.

**[Explore the sample on Hugging Face Datasets](https://huggingface.co/datasets/hzafar/TEXMET)**

## Citation and Credits

### TEXMET Dataset Citation
```bibtex
@dataset{texmet2025,
  title={TEXMET: Curated Textile Images from the Metropolitan Museum of Art},
  author={HAMZA},
  year={2025},
  publisher={Thesis Project},
  version={1.0},
  note={Manually curated dataset: 1,697 objects, 18,644 images}
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
*Last Updated: July 8, 2025*