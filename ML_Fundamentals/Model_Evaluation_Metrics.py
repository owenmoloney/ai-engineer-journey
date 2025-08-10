# Model_Evaluation_Metrics.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def evaluate_model(name, model, X_train, X_test, y_train, y_test, scaler=None):
    # Scale data only for Logistic Regression
    if name == "Logistic Regression" and scaler is not None:
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        X_train_model, X_test_model = X_train, X_test

    # Fit model
    model.fit(X_train_model, y_train)

    # Predictions
    y_pred = model.predict(X_test_model)

    # Some models (e.g., DecisionTree) may not have predict_proba
    try:
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
    except AttributeError:
        # For models without predict_proba, use decision_function or fallback
        if hasattr(model, "decision_function"):
            decision_scores = model.decision_function(X_test_model)
            # Scale to [0,1] using sigmoid approximation
            y_pred_proba = 1 / (1 + np.exp(-decision_scores))
        else:
            # Fallback: use predictions as probabilities (not ideal)
            y_pred_proba = y_pred

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"{name}:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print()

    # Compute ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return fpr, tpr, roc_auc

def main():
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create scaler for logistic regression
    scaler = StandardScaler()

    # Models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    print("Model Evaluation Metrics:\n")

    # Plot ROC curves
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        fpr, tpr, roc_auc = evaluate_model(name, model, X_train, X_test, y_train, y_test, scaler if name == "Logistic Regression" else None)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Models")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
