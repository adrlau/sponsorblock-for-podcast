import pandas as pd
import json
import os
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, WordPunctTokenizer, TweetTokenizer, MWETokenizer, WhitespaceTokenizer, RegexpTokenizer
from collections import Counter
from itertools import repeat, chain
import string 
import multiprocessing
import concurrent.futures
import socket
import torch
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

    

# Create a socket object
s = socket.socket()

# Define the port on which you want to connect
port = 12347

# Bind to the port
s.bind(('', port))
# Put the socket into listening mode
s.listen(5)

while True:
    # Establish a connection with the client
    c, addr = s.accept()

    # Receive data from the client
    data = c.recv(1024)
    #convert data to string
    data = data.decode('utf-8')
    
    # split data by . to get each sentence or at a maximum of 50 characters
    chars_since_last = 0
    sentences = []
    temp = ""
    for char in data:
        # if char == ".":
        #     sentences.append(temp)
        #     temp = ""
        #     chars_since_last = 0
        # elif chars_since_last >= 30:
        if chars_since_last >= 30:
            sentences.append(temp)
            temp = ""
            chars_since_last = 0
        else:
            temp += char
            chars_since_last += 1
    
    for sentence in sentences:
        try:
            # Load the model
            max_token_id = 93821  # Replace with your actual max_token_id
            model = MyModel(max_token_id)
            model.load_state_dict(torch.load('./build/weigths.pth'))

            # Check if GPU is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Convert data to token IDs
            # This will depend on your specific implementation
            df = pd.read_csv(tokens_path)
            token_ids = [string_to_token_ids(sentence, df)]  # replace with your actual conversion function
            
            # probably try to tokenize before splitting on length. 
            X_new = [torch.tensor(x).to(device) for x in token_ids]
            X_new = pad_sequence(X_new, batch_first=True)
            

            # Run the token IDs through the model
            output = model(X_new)
            
            model.eval()
            with torch.no_grad():
                y_new = model(X_new)
                #convert y_new form tensor to float
                y_new = y_new.item() < 0.05
                # Send the result back to the client
                c.send(str("\n\r"+str(y_new)+" : "+str(sentence) + "\n\r").encode())
        except Exception as e:
            c.send(str(f"error {e}\n\r").encode())
            #do the next sentence
            #continue
    c.send(str("done").encode())
    c.close()