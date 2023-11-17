import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk
import spacy
import pandas as pd
import dataconverter3


weights_path = "./build/weigths.pth"
data_path = "./build/data3.csv"

# load data of the form: is_ad,text (text is a arbitrary length string)
print("loading data")
data = pd.read_csv(data_path)

# Drop rows with empty text
print("dropping empty text")
data = data.dropna(subset=['text'])

# split by is_ad
print("splitting data")
data_ad = data[data['is_ad'] == 1]
data_not_ad = data[data['is_ad'] == 0]

# split by train and test
print("splitting data")
train_split = 0.8
train_ad = data_ad[:int(len(data_ad)*train_split)]
test_ad = data_ad[int(len(data_ad)*train_split):]
train_not_ad = data_not_ad[:int(len(data_not_ad)*train_split)]
test_not_ad = data_not_ad[int(len(data_not_ad)*train_split):]

#concatenate the ad and not ad data
print("concatenating data")
train = pd.concat([train_ad, train_not_ad])
test = pd.concat([test_ad, test_not_ad])



import spacy
import multiprocessing
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    temp = [token.text for token in nlp(text)]
    print(temp[:10])
    return temp

# Create a pool of processes
pool = multiprocessing.Pool()

# Apply tokenization function to each text in parallel
print("Tokenizing text")
train['text'] = pool.map(tokenize_text, train['text'])
test['text'] = pool.map(tokenize_text, test['text'])
pool.close()
pool.join()

print(train.head(10))



class LSTM(nn.Module):
    def __init__(self, dimension=128):
        super(LSTM, self).__init__()
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(dimension*2, 1)
        
    def forward(self, x, len):
        x = pack_padded_sequence(x, len, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drop(x)
        x = self.fc(x)
        return x
    