import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Select relevant columns
df = titanic[['sex', 'age', 'fare', 'embarked', 'survived']].copy()

# Impute missing 'age' with median
age_imputer = SimpleImputer(strategy='median')
df['age'] = age_imputer.fit_transform(df[['age']])

# Impute missing 'embarked' with most frequent
embarked_imputer = SimpleImputer(strategy='most_frequent')
df['embarked'] = embarked_imputer.fit_transform(df[['embarked']]).ravel()

# Bin 'age' into categories
df['age_bin'] = pd.cut(df['age'], bins=[0, 12, 20, 40, 60, 120], labels=[0,1,2,3,4])

# Bin 'fare' into quartiles
df['fare_bin'] = pd.qcut(df['fare'], q=4, labels=[0,1,2,3])

# Encode categorical variables
le_sex = LabelEncoder()
df['sex_enc'] = le_sex.fit_transform(df['sex'])

le_embarked = LabelEncoder()
df['embarked_enc'] = le_embarked.fit_transform(df['embarked'])

# Prepare feature matrix X and target y
X = df[['sex_enc', 'age_bin', 'fare_bin', 'embarked_enc']].astype(int)
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Define classifiers
dt = DecisionTreeClassifier(random_state=41)
rf = RandomForestClassifier(random_state=41)

# Parameter grids for GridSearch
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10]
}

rf_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV for Decision Tree
dt_grid = GridSearchCV(dt, dt_params, cv=3, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)

print("Best Decision Tree Params:", dt_grid.best_params_)
dt_best = dt_grid.best_estimator_

y_pred_dt = dt_best.predict(X_test)
print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# GridSearchCV for Random Forest
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("\nBest Random Forest Params:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_

y_pred_rf = rf_best.predict(X_test)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# === Decision Tree Visualization ===
plt.figure(figsize=(20,12))
plot_tree(
    dt_best,
    feature_names=['sex_enc', 'age_bin', 'fare_bin', 'embarked_enc'],
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Best Decision Tree Visualization")
plt.show()
