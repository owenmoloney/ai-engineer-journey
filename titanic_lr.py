import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Select features + target, drop rows with missing values
df = titanic[['sex', 'fare', 'survived']].dropna()

# Encode 'sex' column to numeric
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0

# Define features and target
X = df[['sex', 'fare']]
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Create and fit logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
