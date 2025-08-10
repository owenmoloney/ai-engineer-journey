Week 1 — ML Fundamentals Refresh
Day 1: Intro to Machine Learning
Topics:

Supervised Learning: Learning from labeled data. The model maps inputs to known outputs.

Unsupervised Learning: Finding patterns in unlabeled data. No predefined outputs.

Reinforcement Learning: Learning by interacting with an environment, receiving rewards or penalties.

Logic:

Supervised learns from examples, unsupervised finds hidden structure, reinforcement learns from feedback.

Key Terms:

Label — known outcome for each data point in supervised learning

Feature — input variables describing data points

Reward — feedback signal in reinforcement learning

YouTube: Insert intro video on types of ML

Task: Write a .md explaining each type with examples.

Day 2: Linear Regression
Topics:

Linear Regression: Predicting a continuous output using a linear function of inputs.

Least Squares: Method to find the best fit line minimizing squared errors.

Coefficients: Weights for each feature indicating its influence.

Logic:

Model fits a line that minimizes the distance between predicted and actual values.

Key Terms:

Dependent variable (target) — what you want to predict

Independent variables (features) — predictors

Residual — difference between predicted and actual values

YouTube: Insert linear regression tutorial

Task: Implement linear regression on synthetic data.

Day 3: Logistic Regression (Binary Classification)
Topics:

Logistic Regression: Classification algorithm predicting probabilities between 0 and 1.

Sigmoid Function: Converts linear outputs to probabilities.

Decision Boundary: Threshold to classify outputs as one class or the other.

Logic:

Instead of fitting a line, fits an S-shaped curve to model probability of class membership.

Key Terms:

Odds & log-odds — ratio of probabilities, transformed in logistic regression

Binary outcome — yes/no, spam/ham, etc.

Threshold — cutoff for classification

YouTube: Insert logistic regression explanation

Task: Build spam vs. ham text classifier.

Day 4: Decision Trees & Random Forests
Topics:

Decision Tree: Tree-like model of decisions based on feature splits.

Random Forest: Ensemble of many trees to improve accuracy and reduce overfitting.

Logic:

Trees split data based on feature thresholds to classify or predict. Forests aggregate many trees’ results.

Key Terms:

Node — point where data splits

Leaf — final classification or prediction

Overfitting — model learns noise, not general patterns

YouTube: Insert decision trees and random forests video

Task: Train and compare models on Titanic dataset.

Day 5: Gradient Boosting (XGBoost / LightGBM)
Topics:

Gradient Boosting: Ensemble method where trees are built sequentially to correct errors of previous trees.

XGBoost / LightGBM: Efficient implementations with speed and accuracy improvements.

Logic:

Models “boost” performance by focusing on difficult cases iteratively.

Key Terms:

Weak learner — a model slightly better than random

Residuals — errors used to improve next model

Learning rate — step size for each boosting iteration

YouTube: Insert gradient boosting intro video

Task: Build gradient boosting model on classification data.

Day 6: Clustering (k-means) & Dimensionality Reduction (PCA)
Topics:

k-means Clustering: Partition data into k groups by minimizing distance to cluster centers.

PCA: Reduce dimensions by projecting data onto principal components with most variance.

Logic:

Clustering groups similar data without labels; PCA simplifies data while retaining info.

Key Terms:

Centroid — center point of a cluster

Variance — spread of data along components

Eigenvectors / Eigenvalues — define directions and magnitude of principal components

YouTube: Insert k-means and PCA videos

Task: Cluster MNIST digits and visualize with PCA.

Day 7: Model Evaluation Metrics
Topics:

Accuracy, Precision, Recall, F1-Score: Measures to evaluate classification performance.

ROC-AUC: Measures model’s ability to discriminate classes.

Bias-Variance Tradeoff: Balancing underfitting and overfitting.

Logic:

Metrics quantify how well the model performs and generalizes.

Key Terms:

True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)

ROC Curve — tradeoff between TPR and FPR

Bias — error due to oversimplification

Variance — error due to sensitivity to training data

YouTube: Insert evaluation metrics and bias-variance videos

Task: Write a notebook comparing metrics on one dataset.