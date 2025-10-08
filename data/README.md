# Data Management Plan

This document outlines the strategy for managing data in this project.

## Data Pipeline

The data pipeline consists of three stages, corresponding to the subdirectories in this folder:

1.  **Raw Data (`data/raw`)**: This directory is the starting point for our data pipeline. It should contain the original, immutable data files. For this project, this includes `train.csv` and `test.csv` from the Kaggle competition.

2.  **Processed Data (`data/processed`)**: This directory will store the cleaned, transformed, and feature-engineered data. The scripts in `src/data` and `src/features` will take the raw data as input and produce processed data in this directory. This may include:
    *   Handling missing values.
    *   Encoding categorical features.
    *   Scaling numerical features.
    *   Creating new features (feature engineering).

3.  **External Data (`data/external`)**: This directory is for any additional data sources that may be used to enrich the original dataset. For now, this directory is a placeholder.

## Data Versioning

While not yet implemented, we should consider using a data versioning tool like [DVC (Data Version Control)](https://dvc.org/) in the future. This will allow us to track changes in our data and models, ensuring reproducibility.
