import pandas as pd
import json
import os
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from itertools import repeat, chain
import string 
import multiprocessing
import concurrent.futures
import pandas as pd
import ast
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import trainer2
import dataconverter2
nltk.download('wordnet')



def get_input(test=False):
    #get input from user
    inp = ""
    if not test:
        print("Enter a text of file path with text to be evaluated:")
        inp = input()
    # if empty, use default text
    if inp == "":
        inp = "This is a test message. I really like dogs. And over to something else a message from our sponsor. "
    
    #use os to check if the input is a file path
    if os.path.isfile(inp):
        #if it is a file path, read the file and return the text
        file = open(inp, 'r')
        text = file.read()
        file.close()
        print(f"Read text from file: {text[:100]}...")
        return text
    else:
        print(f"Got string: {inp[:100]}...")
        return inp

def main():
    inp = get_input()
    text = dataconverter2.string_to_token_ids(inp)
    text = dataconverter2.normalize_token_length(text)
    ad = trainer2.predict(text)
    print(f"Ad: {ad}")
    
if __name__ == "__main__":
    main()