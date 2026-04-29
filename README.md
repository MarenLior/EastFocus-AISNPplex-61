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
The reference datasets (`1874_continental.csv`, `1874_east_asian.csv`, `continent_3663.csv`, `eastasia_957.csv`) are included in the `data/` folder. 
**Note:** Due to ethical regulations, the real genotypes of specific populations (e.g., SNH and ZHG) cannot be published. A dummy dataset (`sample_demo.csv`) is provided in the `data/` directory to demonstrate how the inference and spatial mapping code works. 

## Requirements
To install the required dependencies:
```bash
pip install -r requirements.txt

## Usage

Before running the analysis, ensure your working directory is the root of the cloned repository.

### Step 0: Data Preparation
If you do not have access to the restricted clinical genotypes (e.g., SNH and ZHG), please generate the synthetic testing dataset first. This ensures the spatial mapping scripts run correctly:
```bash
python generate_demo_data.py

Step 1: Marker Selection & Feature Reduction
Evaluate baseline models and run the SHAP-RFE pipeline to select the optimal core AIM-SNPs.

For the Continental level:
cd 1_marker_selection/continental/
python 01_model_evaluation.py
python 02_shap_rfe.py
cd ../..

For the East Asian subpopulation level:
cd 1_marker_selection/east_asia/
python 01_model_evaluation.py
python 02_shap_rfe.py
cd ../..

Step 2: Ancestry Inference (AutoML Pipeline)
Train the optimized inference models via FLAML and generate advanced confusion matrices and global SHAP summary plots.
cd 2_ancestry_inference/
python automl_pipeline.py
cd ..

Step 3: Spatial Mapping
Project the testing sample data (sample_demo.csv) onto the reference genetic background using PCA and t-SNE, mapped against predictive probabilities.
cd 3_spatial_mapping/
python spatial_projection.py
cd ..


