import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic binary classification data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # make it Nx1
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define Logistic Regression Model using torch.nn
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instantiate model, loss and optimizer
model = LogisticRegressionModel(input_dim=2)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate PyTorch model
model.eval()
with torch.no_grad():
    predicted = (model(X_test_t) > 0.5).float()
    acc = (predicted.eq(y_test_t).sum().item()) / y_test_t.size(0)
print(f"PyTorch Logistic Regression Accuracy: {acc:.4f}")

# Now using scikit-learn for comparison
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_pred)
print(f"scikit-learn Logistic Regression Accuracy: {sk_acc:.4f}")

