import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd

# Define the path to your data
data_path = "./build/data3.csv"

# Load and preprocess your data
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

# Initialize spacy 'en' model
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    return [token.text for token in nlp(text)]

# Tokenize the text
import concurrent.futures

def tokenize_df(df):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        df['text'] = list(executor.map(tokenize_text, df['text']))
    return df

print("Tokenizing text")
train = tokenize_df(train)
test = tokenize_df(test)
print(f"sample text: {train.iloc[0]['text']}")

#create a vocabulary using spacy 
print("creating vocabulary")
from collections import Counter
word_counts = Counter()
for text in train['text']:
    word_counts.update(text)
for text in test['text']:
    word_counts.update(text)
print(f"number of words: {len(word_counts)}")
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['is_ad']
        return text, label

# Create DataLoaders
train_dataset = TextDataset(train)
test_dataset = TextDataset(test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



