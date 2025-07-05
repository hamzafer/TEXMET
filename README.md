# MET Museum Textiles and Tapestries Dataset

## Overview
This dataset contains textile and tapestry objects from the Metropolitan Museum of Art, collected via their public API.

## Success Rate: 99.274%

## API Expected vs Downloaded
- **Textiles**: 33,194 objects (27,271 with images) out of 33,437 expected
- **Tapestries**: 150 objects (102 with images) out of 151 expected  
- **Failed Downloads**: 244 objects (243 textiles + 1 tapestries)

## 🎯 FINAL RESEARCH DATASET
**27,373 objects with images** - This is the main research dataset!
- Located in: `objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json`

## Directory Structure
```
FINAL_CORRECTED_MET_TEXTILES_DATASET/
├── all_objects/                    # Complete collections (all downloaded objects)
├── objects_with_images_only/       # RESEARCH READY - Objects with images only
├── id_lists/                       # Object ID lists for reference
├── metadata/                       # Dataset documentation
└── README.md                       # This file
```

## Detailed Statistics
- **Total Expected from API**: 33,588 objects
- **Total Successfully Downloaded**: 33,344 objects
- **Total with Images**: 27,373 objects
- **Total Failed**: 244 objects
- **Overall Success Rate**: 99.274%

### Breakdown by Category
- **Textiles**: 33,194/33,437 (99.3% success)
- **Tapestries**: 150/151 (99.3% success)

## Files Created: 20250705_230315

### 🎯 RESEARCH READY (Images Only)
- `ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json` - **FINAL RESEARCH DATASET**
- `textiles_with_images_20250705_230315.json` - Textiles with images only
- `tapestries_with_images_20250705_230315.json` - Tapestries with images only

### Complete Collections (All Downloaded)
- `complete_textiles_20250705_230315.json` - All downloaded textiles
- `complete_tapestries_20250705_230315.json` - All downloaded tapestries
- `textiles_tapestries_intersection_20250705_230315.json` - Objects in both categories

### ID Lists (For Reference)
- `all_objects_with_images_ids_20250705_230315.json` - IDs of final research dataset
- `textile_object_ids_20250705_230315.json` - All textile IDs
- `tapestry_object_ids_20250705_230315.json` - All tapestry IDs
- `failed_object_ids_20250705_230315.json` - Failed download IDs

## Usage Notes
- **Use the `objects_with_images_only/` folder for research** - these objects have downloadable images
- Each object contains full MET API metadata
- Failed objects are mostly confirmed 404s (no longer exist in MET collection)
- Image URLs in `primaryImage` field can be downloaded directly

## Data Quality
- 99.274% success rate from API
- All duplicates removed
- Complete provenance tracking
- Ready for academic research

**🎯 Your final research dataset: 27,373 objects with images!**

## Citation and Credits

### TEXMET Dataset Citation
This dataset (TEXMET) is a curated subset of textiles and tapestries from The Metropolitan Museum of Art's Open Access collection, compiled for research purposes. To the best of our knowledge, this is the first comprehensive compilation of MET's textile and tapestry collections in this format. We share it with the research community under the same open access terms.

**If you use this dataset, please cite the original MET source:**

#### Original Data Source:
```bibtex
@misc{MET_OpenAccess_2024,
  author       = {{The Metropolitan Museum of Art}},
  title        = {Open Access Artworks Dataset},
  year         = {2024},
  howpublished = {\url{https://github.com/metmuseum/openaccess}},
  note         = {CC0 public-domain images and metadata; accessed 4 Jul 2025},
}
```


### Acknowledgments
- **Data Source**: The Metropolitan Museum of Art Open Access Initiative
- **License**: CC0 (public domain) - same as original MET data
- **API Access**: MET Museum Collection API
- **Compilation**: Research subset created July 2025

**Note**: This is a research compilation of existing open access data. All rights and credits belong to The Metropolitan Museum of Art. We thank the MET for making their collection freely available for research and education.

Updated: 2025-07-05 23:03:18.722178
