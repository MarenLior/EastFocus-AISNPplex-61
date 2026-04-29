# EastFocus-AISNPplex-61
Code for Explainable AI-driven marker selection and spatial ancestry mapping

## Overview
The data analysis and model construction were completed in Python. It includes:
1. **Marker Selection**: Evaluates 6 mainstream machine learning models (RF, GBM, XGBoost, LightGBM, CatBoost, AdaBoost), uses Bootstrapping for CI, and non-parametric DeLong tests for ROC curve comparisons. Uses SHAP combined with Recursive Feature Elimination (RFE) to reduce and select the optimal core AIM-SNPs set.
2. **Ancestry Inference**: Utilizes Automated Machine Learning (FLAML) to build highly optimized inference models and interpret decision mechanisms using global SHAP values.
3. **Spatial Mapping**: Combines dimensionality reduction algorithms (PCA/t-SNE) with model prediction probabilities for 2D spatial projection mapping of target populations.

## Repository Structure
- `1_marker_selection/`: Code for model evaluation, DeLong test, SHAP feature importance, and RFE. Divided into `continental` and `east_asia` levels.
- `2_ancestry_inference/`: FLAML-based automated machine learning pipeline and advanced evaluation metrics.
- `3_spatial_mapping/`: PCA and t-SNE spatial mapping algorithms combined with predictive probability.
- `data/`: Contains training/testing datasets and a demo testing sample.
- `generate_demo_data.py`: Script to generate the desensitized dummy dataset (`sample_demo.csv`) to ensure code reproducibility while complying with ethical regulations.

## Data Availability
The reference datasets (`1874_continental.csv`, `1874_east_asian.csv`, `continent_3663.csv`, eastasia_957.csv`) are included in the `data/` folder. 
**Note:** Due to ethical regulations, the real genotypes of specific populations (e.g., SNH and ZHG) cannot be published. A dummy dataset (`sample_demo.csv`) is provided in the `data/` directory to demonstrate how the inference and spatial mapping code works. 

## Requirements
To install the required dependencies:
```bash
pip install -r requirements.txt

