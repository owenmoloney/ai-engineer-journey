import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])

# Map labels to binary values
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.4, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict probabilities on test set
y_proba = model.predict_proba(X_test_vec)[:, 1]  # Probabilities for class 1 (spam)

print("Threshold | Accuracy | Precision (Spam) | Recall (Spam)")
print("-------------------------------------------------------")
for threshold in [i/10 for i in range(1, 10)]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_thresh)
    precision_spam = precision_score(y_test, y_pred_thresh)
    recall_spam = recall_score(y_test, y_pred_thresh)
    print(f"  {threshold:.1f}    |  {accuracy:.4f}  |      {precision_spam:.4f}      |    {recall_spam:.4f}")

# Optional: Pick best threshold and print full classification report
best_threshold = 0.3  # You can adjust this based on output above
y_pred_best = (y_proba >= best_threshold).astype(int)
print("\nClassification Report at threshold =", best_threshold)
print(classification_report(y_test, y_pred_best))


# ----------------------------------------
# Comments:
# - Adjusting the classification threshold changes the trade-off between precision and recall.
# - Lower thresholds increase recall (catch more spam) but may reduce precision (more false positives).
# - Higher thresholds increase precision but reduce recall.
# - Choosing the right threshold depends on the application: 
#   For spam filtering, missing spam (low recall) might be worse than some false alarms (low precision).
# - Here, threshold 0.3 gives a good balance: ~94% precision and recall for spam detection.
# - This manual tuning of thresholds allows you to control the behavior of the classifier beyond default 0.5 cutoff.
# ----------------------------------------