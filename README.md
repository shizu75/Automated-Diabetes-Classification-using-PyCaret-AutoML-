# Automated Diabetes Classification using PyCaret (AutoML)

## Project Overview
This project demonstrates the use of **PyCaret**, a low-code **AutoML framework**, to build, compare, tune, and evaluate multiple machine learning models for **diabetes classification**. Instead of manually training individual models, PyCaret automates the entire machine learning pipeline, making model development faster, reproducible, and efficient.

The project showcases an end-to-end **AutoML workflow** including model comparison, selection, hyperparameter tuning, and final evaluation.

---

## Objectives
- Apply AutoML techniques for classification problems
- Automatically compare multiple machine learning models
- Select the best-performing model based on accuracy
- Perform hyperparameter tuning on the best model
- Evaluate final model performance on unseen test data

---

## Technologies Used
- Python 3
- PyCaret
- Pandas
- scikit-learn

---

## Dataset Description
The dataset used is the **Diabetes dataset**, loaded directly using PyCaret’s built-in dataset utility.

### Target Variable
- **Class variable**: Indicates diabetes outcome (binary classification)

### Features
- Multiple numerical medical attributes related to diabetes diagnosis

The dataset is split into training and testing sets to simulate a real-world machine learning workflow.

---

## Data Preparation
- Dataset loaded using `get_data('diabetes')`
- Split into:
  - 70% Training data
  - 30% Testing data
- PyCaret `setup()` function automatically handles:
  - Data preprocessing
  - Feature scaling
  - Missing value handling
  - Encoding (if required)
  - Train-validation splitting

---

## AutoML Workflow

### 1. PyCaret Setup
The `setup()` function initializes the PyCaret environment by defining:
- Dataset
- Target variable
- Preprocessing pipeline

This step prepares the data for automated modeling.

---

### 2. Model Comparison
- `compare_models()` evaluates multiple classification algorithms
- Models are ranked based on performance metrics
- Enables quick identification of strong baseline models

---

### 3. Automatic Model Selection
- `automl(optimize='Accuracy')` selects the best-performing model
- Optimization is done based on **accuracy**
- The selected model is printed for inspection

---

### 4. Hyperparameter Tuning
- `tune_model()` improves the selected model
- Automatically searches for optimal hyperparameters
- Results in a tuned and more robust classifier

---

### 5. Model Prediction
- `predict_model()` generates predictions on unseen test data
- Outputs predicted labels along with confidence scores

---

## Model Evaluation
- Final evaluation performed using **accuracy score**
- Accuracy calculated by comparing:
  - True labels from test data
  - Predicted labels from the tuned model

This provides a reliable estimate of real-world model performance.

---

## Results
- Best classification model automatically selected
- Tuned model achieves improved accuracy
- End-to-end pipeline completed with minimal code
- Demonstrates effectiveness of AutoML for rapid experimentation

---

## How to Run the Project

### Prerequisites
Install required libraries:
- pycaret
- pandas
- scikit-learn

---

### Steps
1. Run the Python script
2. Allow PyCaret to initialize and preprocess data
3. Observe:
   - Model comparison leaderboard
   - Best selected model
   - Tuned model output
   - Predictions on test data
   - Final accuracy score

---

## Learning Outcomes
- Understanding AutoML concepts and workflows
- Experience using PyCaret for rapid model development
- Ability to automate model selection and tuning
- Reduced manual effort in machine learning pipelines

---

## Future Enhancements
- Optimize using different metrics (AUC, F1-score, Recall)
- Perform feature importance analysis
- Add cross-validation control
- Deploy trained model as an API or web application
- Compare AutoML results with manually built models

---

## Use Case
This project is ideal for:
- Machine learning automation demonstrations
- Data science portfolios
- Rapid prototyping of classification models
- Healthcare and medical data analysis use cases

---

## Author
Developed as an educational AutoML project demonstrating diabetes classification using PyCaret.
