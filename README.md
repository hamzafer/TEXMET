# TEXMET: Curated Textile Dataset from the Metropolitan Museum of Art

## Overview
**TEXMET** is a high-quality, manually curated dataset of textile and tapestry objects from the Metropolitan Museum of Art's Open Access collection. This dataset has been carefully cleaned, validated, and optimized for computer vision and deep learning applications.

## Dataset Access
🤗 A sample of the curated dataset is available on Hugging Face.

**[Explore the sample on Hugging Face Datasets](https://huggingface.co/datasets/hzafar/TEXMET)**

## TEXMET FINAL - CURATED DATASET
- **Total Images**: 18,644 high-resolution images
- **Unique Objects**: 1,697 textile/tapestry objects  
- **Average Resolution**: 3.3 MP (1557 x 1631 pixels)
- **Average File Size**: 1.35 MB per image
- **Format**: 100% JPEG, 92% RGB mode
- **Quality**: Hand-selected, duplicates removed, visually validated

## Dataset Statistics
- **Curation Method**: Manual validation using FiftyOne + CLIP embeddings
- **Original Collection**: 27,373 objects → **Curated to 18,644 premium samples**
- **Quality Control**: Similarity analysis, duplicate removal, visual inspection
- **Coverage**: Global textile traditions spanning 4000+ years
- **Image Quality**: High-resolution, suitable for ML/CV applications

## Directory Structure
```
TEXMET/
├── data/                     # All data-related files and scripts
│   ├── clean_dataset/        # --> The final, curated TEXMET dataset
│   ├── download/             # Scripts and logs for downloading data
│   ├── processing/           # Data processing and cleaning scripts
│   └── ...                   # (bad_dataset, raw data, etc.)
│
├── experiments/              # Self-contained research experiments
│   ├── inpainting-exp/       # Image inpainting experiments
│   ├── sam/                  # Segment Anything Model experiments
│   └── thread_count_analysis/ # Scripts for the thread count analysis
│
├── visuals/                  # Visualization scripts and results
│   └── osebergvisuals/
│
├── logs/                     # Log files from various processes
├── website/                  # Code for the project website/filter
└── README.md                 # Project overview
```

## Data Quality & Curation Process
1. **Initial Collection**: 27,373 objects from MET API (99.274% success rate)
2. **Filtering**: Focus on high-quality textile/tapestry objects
3. **Manual Curation**: Visual inspection and validation using FiftyOne
4. **Duplicate Removal**: CLIP embedding analysis + manual verification
5. **Quality Control**: Removed low-quality, irrelevant, or damaged images
6. **Final Validation**: **1,697 premium objects** with **18,644 images**

## Image Specifications
- **Total Images**: 18,644 images
- **Average Dimensions**: 1557 x 1631 pixels
- **Average Aspect Ratio**: 1.14 (slightly portrait)
- **Average File Size**: 1.35 MB
- **Average Resolution**: 3.3 megapixels
- **Format**: 100% JPEG
- **Color Mode**: 92% RGB, 8% other modes
- **Quality**: High-resolution, ML/CV ready

## Research Applications
- ✅ Textile pattern recognition and classification
- ✅ Cultural heritage digitization and analysis
- ✅ Style transfer and generative AI
- ✅ Historical textile dating and attribution
- ✅ Cross-cultural design studies
- ✅ Computer vision algorithm development
- ✅ Digital art and design inspiration
- ✅ Large-scale image analysis and clustering

## Usage Examples
```python
import json
import pandas as pd

# Load the curated dataset
with open('data/clean_dataset/clean_textiles_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"TEXMET Final: {len(df)} curated objects")

# Load metadata
with open('data/clean_dataset/texmet_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f"Total images: {metadata['image_statistics']['total_images']}")
```

## Version History
- **TEXMET Final v1.0** (July 2025): **1,697 curated objects, 18,644 images**
- **Raw Collection** (July 2025): 27,373 objects collected from MET API

## Technical Specifications
- **Format**: JSON with UTF-8 encoding
- **Image Format**: 100% JPEG
- **Resolution**: High-resolution (average 3.3 MP)
- **File Organization**: Git LFS for large files
- **Compatibility**: Ready for PyTorch, TensorFlow, OpenCV

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
This dataset (TEXMET) is a curated subset of textiles and tapestries from The Metropolitan Museum of Art's Open Access collection, compiled for research purposes. To the best of our knowledge, this is the first comprehensive compilation of MET's textile and tapestry collections in this format, manually curated and optimized for computer vision applications.

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
- **License**: CC0 (Public Domain) - same as original MET data
- **Usage Rights**: Free for any purpose without restriction
- **Attribution**: Metropolitan Museum of Art (recommended)
- **Copyright**: All images are in the public domain

## Acknowledgments
- **Data Source**: The Metropolitan Museum of Art Open Access Initiative
- **Curation Tools**: FiftyOne, CLIP embeddings, manual validation
- **API Access**: MET Museum Collection API
- **Project**: TEXMET - Textile Analysis Thesis Project
- **Created**: July 2025

## Performance & Quality Metrics
- **Image Quality**: 100% JPEG, high resolution
- **Duplicate Removal**: CLIP similarity + manual verification
- **Manual Validation**: Visual inspection of all selected objects
- **Research Ready**: Optimized for ML/CV applications

**Note**: This is a research compilation and curation of existing open access data. All rights and credits belong to The Metropolitan Museum of Art. We thank the MET for making their collection freely available for research and education.

---
**🎯 TEXMET Final: 1,697 Premium Objects, 18,644 High-Quality Images**
Total Dataset Size: 26.5 GB.
**Ready for Computer Vision and Machine Learning Applications!**

*Last Updated: July 8, 2025*
