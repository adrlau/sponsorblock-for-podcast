import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Define the model
class MyModel(nn.Module):
    def __init__(self, max_token_id):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_token_id + 1, embedding_dim=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)
    
    
    
# to test on strings

import pandas as pd
import json
import os
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, WordPunctTokenizer, TweetTokenizer, MWETokenizer, WhitespaceTokenizer, RegexpTokenizer

from collections import Counter
from itertools import repeat, chain
import string 
import multiprocessing
import concurrent.futures

transcriptions_path = "./build/transcripts.json"
tokens_path = "./build/tokens.csv"
data_path = "./build/data.csv"
data_max_tokens = 50
tokenizer = TreebankWordTokenizer()
def stringparser(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace unwanted characters with appropriate substitutes
    replacements = {
        '"': '',
        '\'': '',
        '`': '',
        '´': '',
        '.': ' . ',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
        "": '',
        "": '',
        "  ": ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove non-printable characters
    text = ''.join(char for char in text if char in string.printable)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text


def string_to_token_ids(string, token_dataframe):
    string = stringparser(string)
    tokens = tokenizer.tokenize(string)
    
    #convert tokens to list of ids in df using the tokens in the df as a reference. If a token is not in the df, add it to the df convert it to single letter token and add it to the list of ids
    ids = []
    for token in tokens:
        #find id of the text in the df
        id = token_dataframe[token_dataframe['text'] == token].index.values
        if id.size == 0:
            # split token into single letters and find their ids
            for char in token:
                id = token_dataframe[token_dataframe['text'] == char].index.values
                # if the char is not in the df, skip it
                if id.size == 0:
                    continue
                ids += [id[0]]
        else :
            ids += [id[0]]
    return ids
    
    
    
    
    

# Load the weights
# max_token_id = 93821  # Replace with your actual max_token_id
max_token_id = 93821  # Replace with your actual max_token_id
model = MyModel(max_token_id)
model.load_state_dict(torch.load('./build/weigths.pth'))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ad_test = string_to_token_ids("this is a test", pd.read_csv(tokens_path))
ad_test1 = string_to_token_ids("Have you heard about Octa, The free a", pd.read_csv(tokens_path))
ad_test2 = string_to_token_ids("With all the recent news about online security breaches, it's very hard not to worry about where your data goes. Making an online purchase or simply accessing your email could put your private information at risk ... That's why I decided to take back my privacy using ExpressVPN. ExpressVPN has easy-to-use apps and runs seamlessly in the background of my computer phone tablet.", pd.read_csv(tokens_path))
ad_test3 = string_to_token_ids("With all the recent news about online security breaches, it's very hard not", pd.read_csv(tokens_path))
ad_test4 = string_to_token_ids("Ministers have formally requested that a pro-Palestinian march being planne", pd.read_csv(tokens_path))
ad_test5 = string_to_token_ids("And now a message from our sponsors.", pd.read_csv(tokens_path))
ad_test6 = string_to_token_ids("this mobile", pd.read_csv(tokens_path))
ad_test7 = string_to_token_ids("We're gonna find out for science", pd.read_csv(tokens_path))#from traiining
ad_test8 = string_to_token_ids("Maybe this idea isn't that crazy, you know gaming on a 64 core CPU because believe it or not", pd.read_csv(tokens_path))#from training should be considered sponsor as false positive.
ad_test9 = string_to_token_ids("I've covered in detail before.", pd.read_csv(tokens_path))


# Define your new data
# new_data = [ad_test, ad_test1, ad_test2, ad_test3]
new_data = [ad_test9]

# Prepare the new data
X_new = [torch.tensor(x).to(device) for x in new_data]
X_new = pad_sequence(X_new, batch_first=True)

# Use the model to classify the new data
model.eval()
with torch.no_grad():
    y_new = model(X_new)
    print(y_new)



# # evaluate the model on the dataset
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import ast
# df = pd.read_csv('./build/data-save-1.csv', converters={'token_ids': ast.literal_eval})
# #drop rows with token_ids of length 1 or less
# df = df[df['token_ids'].map(len) > 1]
# # remove entries where is_ad is 0
# # df = df[df['is_ad'] == 1]
# class MyDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = [x.to(device) for x in X]
#         self.y = y.to(device)

#     def __getitem__(self, i):
#         return self.X[i], self.y[i]

#     def __len__(self):
#         return len(self.y)
# # Prepare the data
# X = [torch.tensor(x) for x in df['token_ids'].tolist()]
# y = torch.tensor(df['is_ad'].tolist())

# # Pad sequences for consistent input size
# X = pad_sequence(X, batch_first=True)

# # Create DataLoader
# dataset = MyDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=32)

# correct_predictions = 0
# total_predictions = 0

# threshold = 0.109

# model.eval()
# with torch.no_grad():
#     for X_batch, y_batch in dataloader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         y_pred = model(X_batch)
#         predicted = (y_pred > threshold).int()
        
#         correct_predictions += (predicted == y_batch).all(dim=1).sum().item()
#         total_predictions += y_batch.size(0)
#         print(f"Correct predictions: {correct_predictions}, Total predictions: {total_predictions}, Accuracy: {correct_predictions / total_predictions * 100:.2f}%")

# accuracy = correct_predictions / total_predictions
# print(f'Accuracy: {accuracy * 100:.2f}%')


