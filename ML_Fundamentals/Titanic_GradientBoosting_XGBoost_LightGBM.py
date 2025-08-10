import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Fill missing values for relevant columns
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['fare'].fillna(titanic['fare'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Create new features
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Encode categorical variables
# We'll use pandas get_dummies for simplicity
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone'], drop_first=True)

# Drop columns that won't be used or have too many missing values
titanic.drop(columns=['alive', 'adult_male_False', 'deck_nan'], errors='ignore', inplace=True)  # adjust as needed

# Drop rows with any remaining missing values to keep it clean
titanic.dropna(inplace=True)

# Define features and target
X = titanic.drop(columns=['survived'])
y = titanic['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use GradientBoostingClassifier from sklearn
gb = GradientBoostingClassifier(random_state=42)

# Grid search for best params (optional, smaller grid for speed)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)

# Predict and evaluate
y_pred = grid_search.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))
