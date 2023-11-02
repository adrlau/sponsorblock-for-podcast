import torch
import torch.nn as nn
import torch.optim as optim
import json


# example of a transcript
# # Load the data from transcripts.json
with open('build/transcripts.json', 'r') as f:
    data = json.load(f)

# Define the model
class AdDetector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdDetector, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.sigmoid(self.hidden(x))
        output = self.sigmoid(self.out(hidden))
        return output

# Define the hyperparameters
input_size = len(data[0]['transcript'])
hidden_size = 32
output_size = 1
lr = 0.01
epochs = 100

# Initialize the model and optimizer
model = AdDetector(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# Train the model
for epoch in range(epochs):
    for d in data:
        x = torch.tensor(d['transcript'], dtype=torch.float32)
        y = torch.tensor([d['is_ad']], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'ad_detector.pt')