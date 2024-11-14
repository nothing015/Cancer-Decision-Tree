# Optimizing Decision Tree Classifiers for Breast Cancer Diagnosis with Cross-Validation and Hyperparameter Tuning
This project uses decision tree classifiers to predict breast cancer diagnosis outcomes. It employs the Breast Cancer Wisconsin dataset from Scikit-Learn and explores model optimization through hyperparameter tuning and cross-validation.
## Project Overview
The goal of this project is to:

Build a decision tree classifier for diagnosing breast cancer using the Wisconsin dataset.
Optimize the model using hyperparameter tuning (maximum tree depth and minimum samples to split).
Visualize the impact of hyperparameters on training and test accuracy to select the best model.
Implement cross-validation techniques for robust model performance estimation.
## Dataset
The Breast Cancer Wisconsin dataset, available in Scikit-Learn, contains 569 instances and 30 features describing tumor properties, labeled as either benign or malignant.

Features: 30 attributes related to cell nuclei measurements in digitized images of fine needle aspirates (e.g., radius, texture, perimeter, area).
Target: Binary classification (0: Malignant, 1: Benign).
For more details, refer to the Scikit-Learn dataset documentation.

## Project Structure
Part 1: Load and preprocess the dataset. Split it into training (60%) and testing (40%) sets.
Part 2: Train and evaluate the decision tree model using entropy as the criterion. Visualize classifier accuracy across tree depths.
Part 3: Apply GridSearchCV to identify optimal hyperparameters and visualize the final model.
