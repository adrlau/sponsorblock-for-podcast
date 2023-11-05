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

transcriptions_path = "./build/transcripts.json"
tokens_path = "./build/tokens.csv"
data_path = "./build/data.csv"
port = 12347 #port to listen on
max_token_id = 93821  # Replace with your actual max_token_id (printed at start of training)
data_max_tokens = 8
tokenizer = TreebankWordTokenizer()
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
def stringparser(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace unwanted characters with appropriate substitutes
    replacements = {
        '"': '',
        "'": '',
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

def token_ids_to_string(ids):
    try:
        df = pd.read_csv(tokens_path)
        string = ""
        for id in ids:
            string += df['text'][id] + " "
    except Exception as e:
        string = "from token_ids_to_string" + str(e) + "  tokens:" + str(ids)
    return string

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


def load_model():
    model = MyModel(max_token_id)
    model.load_state_dict(torch.load('./build/weigths-save1.pth'))
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def create_socket():
    s = socket.socket()
    s.bind(('', port))
    s.listen(1) #max 1 connection
    c, addr = s.accept()
    return s, c, addr


def main():
    while True:
        s, c, addr = create_socket()
        # Receive data from the client
        data = c.recv(2048).decode('utf-8')
        #remove all linebreaks and tabs
        data = data.replace("\n", " ")
        data = data.replace("\r", " ")
        data = data.replace("\t", " ")
        
        #convert data to list tokens
        tokens = string_to_token_ids(data, pd.read_csv(tokens_path))
        print(len(tokens))
        #split tokens into lists of max length
        tokens = [tokens[i:i + data_max_tokens] for i in range(0, len(tokens), data_max_tokens)]
        print(len(tokens))
        
        # for each list of tokens
        for token_ids in tokens:
            try:
                # load model
                model,device = load_model()
                #convert to padded tensor
                X_new = pad_sequence([torch.tensor(token_ids)], batch_first=True).to(device)
                text = token_ids_to_string(token_ids)
                print(text)
                # Run the token IDs through the model
                output = model(X_new)
                
                #convert output to float with 4 decimals
                output = round(output.item(), 4)
                output = "{:4}".format(output)
                # Send the result back to the client
                c.send(str("\n\r"+str(output)+" : "+text+ "\n\r").encode())
                
            except Exception as e:
                c.send(str(f"error {e}\n\r").encode())
                #do the next sentence
                #continue
        c.send(str("done").encode())
        c.close()

while True:
    try:
        main()
    except Exception as e:
        print(e)
        #sleep for 5 seconds
        import time
        time.sleep(1)
        continue