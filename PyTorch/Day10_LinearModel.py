import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear regression model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input dim=1, output dim=1

    def forward(self, x):
        return self.linear(x)

# Create model instance
model = LinearModel()

# Example data: y = 2x + 1
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, test the model
model.eval()
test_input = torch.tensor([[5.0]])
predicted = model(test_input).item()
print(f'Prediction for input 5.0: {predicted:.2f}')

