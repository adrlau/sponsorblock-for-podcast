import pandas as pd
import ast

# Load the data of format.
# is_ad,token_ids
# 1,"[13, 105, 11, 844, 64, 2219, 5]"
# 0,"[49, 7141, 15641, 7, 1, 5733, 2723, 2, 0, 2171, 3686, 7]"
# 0,"[8, 133, 124, 1, 36, 3, 105, 50, 227, 5, 68, 7, 9, 483, 25, 10]"
# 0,[63]
df = pd.read_csv('./build/data-save-1.csv', converters={'token_ids': ast.literal_eval})
#drop rows with token_ids of length 1 or less
df = df[df['token_ids'].map(len) > 1]

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x) for x in X]
        self.y = torch.tensor(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

# Prepare the data
X = [torch.tensor(x) for x in df['token_ids'].tolist()]
y = torch.tensor(df['is_ad'].tolist())

# Pad sequences for consistent input size
X = pad_sequence(X, batch_first=True)

# Create DataLoader
dataset = MyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)


# get the number of token_id lists
max_token_id = max([max(x) for x in df['token_ids'].tolist()])
print(max_token_id)


# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_token_id + 1, embedding_dim=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

model = MyModel()

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    print(f"epoch {epoch}")
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch.float())
        loss.backward()
        optimizer.step()

# Save the weights
torch.save(model.state_dict(), './build/weigths.pth')