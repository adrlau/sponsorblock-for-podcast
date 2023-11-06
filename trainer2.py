import pandas as pd
import ast
import torch
import numpy as np
import copy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import dataconverter2

weights_path = "./build/weigths.pth"
data_path = "./build/data.csv"
feature_size = dataconverter2.data_max_tokens #number of tokens in each sample of data currently 16
batch_size = 32
epochs = 50
lr = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(15856726355239440931)
#print torch seed
print(f"torch seed: {torch.initial_seed()}")

def get_data():
    #dataframe with all tokens and their ids 
    #is_ad,token_ids
    #1,"[534, 293, 1782, 1783, 392, 116, 244, 1784, 1785, 1786, 3, 688, 726, 725, 1787, 1788]"
    df = pd.read_csv(data_path, converters={'token_ids': ast.literal_eval}) #load data from csv file and convert token_ids column from string to list
    #randomize the order of the samples in the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    print(f"{len(df)} samples in dataset")
    print(df["token_ids"].values)
    
    X = torch.tensor(df['token_ids'].tolist(), dtype=torch.float32)
    y = torch.tensor(df['is_ad'].values, dtype=torch.float32).reshape(-1, 1)
    # split data into train and test
    split = int(len(X)*0.9)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test

class Classificator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(feature_size, feature_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(feature_size, feature_size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(feature_size, feature_size)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(feature_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x    
    

def model_train(model, X_train, y_train, X_test, y_test):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch}")
        for start in range(0, len(X_train), batch_size):
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            
            # acc = (y_pred.round() == y_batch).float().mean()
            
            #Use test data to evaluate model accuracy
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                loss = loss_fn(y_pred, y_test)
                acc = (y_pred.round() == y_test).float().mean() 
            
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        print(f"Loss: {float(loss)}, Accuracy: {float(acc)}")
        
    #save weigths
    print(f" Best accuracy: {best_acc}, saving weights to {weights_path}")
    torch.save(best_weights, weights_path)

def predict(inp):
    # Load model with weights from file
    model = Classificator().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
    
    # Convert input to tensor
    inp_tensor = torch.tensor(inp, dtype=torch.float32).to(device)
    # Make prediction
    with torch.no_grad():
        output = model(inp_tensor)
    #convert output to list of floats rounded to 4 decimals
    output = output.tolist()
    output = [round(x[0], 4) for x in output]
    return output

if __name__ == "__main__":
    X,y,Xe,ye = get_data()
    model = Classificator().to(device)
    print(f" parameters:{sum([x.reshape(-1).shape[0] for x in model.parameters()])}") 
    model_train(model,X,y, Xe, ye)
    test = dataconverter2.string_to_token_ids("This is a test. Is there an advert in here? I really like dogs. This is a message from our sponsor")
    test = dataconverter2.normalize_token_length(test)
    print(f"{predict(test)}")
