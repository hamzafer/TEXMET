# TeXMET: Curated Textile Dataset from the Metropolitan Museum of Art

## Overview
**TeXMET** is a high-quality, manually curated dataset of textile and tapestry objects from the Metropolitan Museum of Art's Open Access collection. This dataset has been carefully cleaned, validated, and optimized for computer vision and machine learning applications.

## 🎯 TEXMET FINAL - CURATED DATASET
**1,697 premium quality objects** - Manually curated and validated!
- **Total Images**: 18,644 high-resolution images
- **Unique Objects**: 1,697 textile/tapestry objects  
- **Average Resolution**: 3.3 MP (1557 x 1631 pixels)
- **Average File Size**: 1.35 MB per image
- **Format**: 100% JPEG, 92% RGB mode
- **Quality**: Hand-selected, duplicates removed, visually validated

## Dataset Statistics
- **Curation Method**: Manual validation using FiftyOne + CLIP embeddings
- **Original Collection**: 27,373 objects → **Curated to 1,697 premium objects**
- **Quality Control**: Similarity analysis, duplicate removal, visual inspection
- **Coverage**: Global textile traditions spanning 4000+ years
- **Image Quality**: High-resolution, suitable for ML/CV applications

## Directory Structure
```
TEXMET/
├── clean_dataset/                   # 🎯 TEXMET FINAL CURATED DATASET
│   ├── clean_textiles_dataset.json  # Main dataset (1,697 objects)
│   ├── texmet_metadata.json         # Comprehensive metadata
│   ├── images/                      # Primary images (~1,697 images)
│   └── additional_images/           # Additional views (~17K images)
├── FINAL_CORRECTED_MET_TEXTILES_DATASET/  # Original raw data (27,373 objects)
│   └── objects_with_images_only/    # Complete raw collection
└── README.md                        # This file
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
with open('clean_dataset/clean_textiles_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"TeXMET Final: {len(df)} curated objects")

# Load metadata
with open('clean_dataset/texmet_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f"Total images: {metadata['image_statistics']['total_images']}")
```

## Version History
- **TeXMET Final v1.0** (July 2025): **1,697 curated objects, 18,644 images**
- **Raw Collection** (July 2025): 27,373 objects collected from MET API

## Technical Specifications
- **Format**: JSON with UTF-8 encoding
- **Image Format**: 100% JPEG
- **Resolution**: High-resolution (average 3.3 MP)
- **File Organization**: Git LFS for large files
- **Compatibility**: Ready for PyTorch, TensorFlow, OpenCV

## Citation and Credits

### TeXMET Dataset Citation
```bibtex
@dataset{texmet2025,
  title={TeXMET: Curated Textile Images from the Metropolitan Museum of Art},
  author={HAMZA},
  year={2025},
  publisher={Thesis Project},
  version={1.0},
  note={Manually curated dataset: 1,697 objects, 18,644 images}
}
```

### Original Data Source
This dataset (TeXMET) is a curated subset of textiles and tapestries from The Metropolitan Museum of Art's Open Access collection, compiled for research purposes. To the best of our knowledge, this is the first comprehensive compilation of MET's textile and tapestry collections in this format, manually curated and optimized for computer vision applications.

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
- **Project**: TeXMET - Textile Analysis Thesis Project
- **Created**: July 2025

## Performance & Quality Metrics
- **Curation Ratio**: 1,697/27,373 = 6.2% (highly selective)
- **Image Quality**: 100% JPEG, high resolution
- **Duplicate Removal**: CLIP similarity + manual verification
- **Manual Validation**: Visual inspection of all selected objects
- **Research Ready**: Optimized for ML/CV applications

**Note**: This is a research compilation and curation of existing open access data. All rights and credits belong to The Metropolitan Museum of Art. We thank the MET for making their collection freely available for research and education.

---
**🎯 TeXMET Final: 1,697 Premium Objects, 18,644 High-Quality Images**
**Ready for Computer Vision and Machine Learning Applications!**

*Last Updated: July 8, 2025*
