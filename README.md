# Fake Review Detection (Yelp)

Machine-learning project exploring how to classify **fraudulent versus genuine Yelp reviews** using review-text, reviewer-behavior, and product-level signals.

## Approach

- Extract **17 engineered features**, including polarity, subjectivity, word count, rating variance, content similarity, review frequency, singleton status, and positive/negative review ratios.
- Compare classical supervised classifiers (**Random Forest, Naive Bayes, Decision Tree, XGBoost**) with an **Artificial Neural Network (ANN)**.
- Evaluate with train/test splits and stratified k-fold cross-validation, and explore permutation-based feature importance with **ELI5**.
- The ANN experiments reported approximately **75% accuracy**. This is a previously reported project result and has not been independently reproduced for this publication.

## Notebooks

| File | Purpose |
| --- | --- |
| `notebooks/Feature_Extraction.ipynb` | Review and metadata feature extraction |
| `notebooks/SWM_UserFeatureExtraction_ModelTraining.ipynb` | User features and Random Forest / Naive Bayes training |
| `notebooks/Product_features_and_content_similarity_and_model_train.ipynb` | Product features, TF-IDF/content similarity, and model training |
| `notebooks/ANN.ipynb` | Artificial neural network modeling |
| `notebooks/XGB&DecisionTree.ipynb` | XGBoost and Decision Tree experiments |
| `notebooks/Feature_Importance.ipynb` | ELI5/permutation feature importance |
| `notebooks/Visualization.ipynb` | Dataset and result visualization |

## Run locally

1. Use a Python/Jupyter environment and install the packages in `requirements.txt` (package versions may need adjustment for modern environments).
2. Obtain the Yelp datasets and prepare the raw metadata/reviews and generated `featuresAll.csv`, `labels.csv`, and related intermediate files. **Data files are not included in this code-only publication.**
3. Open the notebooks in Jupyter/Colab. Some notebook cells use historic absolute Google Colab/Drive paths or expect inputs in the working directory; adjust those paths before running.

> **Note:** This is a cleaned, code-focused version of the project. Saved outputs were removed to keep the code browsable and avoid committing large embedded outputs. Core notebook source cells were preserved. This should not be presented as a tested, one-command reproducible pipeline.

## Code provenance

Original shared project repository: [pinakiweb/fake-review-detection](https://github.com/pinakiweb/fake-review-detection).
