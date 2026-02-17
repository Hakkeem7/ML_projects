# Breast Cancer Prediction using SVM (Support Vector Machine):

This project is a Machine Learning classification system that predicts whether a breast tumor is Malignant (Cancerous) or Benign (Non-cancerous) using the Breast Cancer Wisconsin Dataset.

The model is built using Support Vector Machine (SVC) and improved using GridSearchCV hyperparameter tuning.
The project also includes PCA (2D reduction) for visualization and decision boundary plotting.

## Project Overview:

Breast cancer is one of the most common cancers in the world.
Early detection plays a major role in saving lives.

This project uses supervised learning to classify tumors based on medical features such as:

Radius

Texture

Smoothness

Concavity

Symmetry

And more...

## Objective:

Build a machine learning model to classify breast tumors

Train and test the model using real dataset

Improve model performance using GridSearchCV

Visualize classification boundary using PCA

## Algorithms & Techniques Used:
### Support Vector Machine (SVM)

Model used: SVC (Support Vector Classifier)

Kernels used:

Linear Kernel

RBF Kernel

### Hyperparameter Tuning:

Used GridSearchCV

Tuned parameters:

C

gamma

kernel

### PCA (Principal Component Analysis):

Reduced dataset to 2 dimensions

Used for visualization and decision boundary plot

## Dataset:

Dataset used: Breast Cancer Wisconsin Dataset
Loaded from:

from sklearn.datasets import load_breast_cancer

### Target Classes:

0 → Malignant

1 → Benign

## Project Structure:
Breast-Cancer-Prediction/
│── breast_can_pred.ipynb
│── README.md

## Technologies Used:

Python 

Jupyter Notebook

NumPy

Pandas

Matplotlib

Scikit-learn

## Model Evaluation:

The model is evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

## Results:

The optimized SVM model gives good accuracy and successfully predicts whether the tumor is:

Benign
or
Malignant
