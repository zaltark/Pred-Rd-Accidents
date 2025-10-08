# Model Management Plan

This document outlines the strategy for managing models in this project.

## Model Training

Models will be trained using the script located at `src/models/train_model.py`. This script will handle:

*   Loading the processed training data.
*   Initializing the model with parameters from `config/config.py`.
*   Training the model on the data.
*   Evaluating the model using the specified metric (RMSE).
*   Saving the trained model to this directory.

## Model Naming and Versioning

To ensure we can track and manage different versions of our models, we will use a consistent naming convention for saved model files:

`<model_name>_<timestamp>.pkl`

For example: `random_forest_20251008123000.pkl`

This convention will help us identify the model type and when it was trained.

## Model Storage

Trained and serialized models will be stored in this `models/` directory. These models are the output of the training process and can be loaded by the prediction script (`src/models/predict_model.py`) to make predictions on new data.

## Experiment Tracking

For initial development, we will log model performance and parameters to the console and log files. As the project matures, we should consider integrating a more advanced experiment tracking tool like [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/). These tools will allow us to:

*   Log and compare model parameters and metrics across experiments.
*   Store model artifacts in a more organized way.
*   Manage the entire model lifecycle from experimentation to deployment.
