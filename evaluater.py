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

threshold = 0.6

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

def is_ad_text(text):
    #convert text to token ids
    text = dataconverter2.string_to_token_ids(text)
    #predict if the text is an ad
    ad = trainer2.predict(text)
    #if the text is an ad, return true
    if ad[0] > threshold:
        return True
    else:
        return False

def is_ad_tokens(tokens):
    tokens = dataconverter2.normalize_token_length(tokens)
    #predict if the text is an ad
    ad = trainer2.predict(tokens)
    
    #return the ad values for each token instead of just the first one
    

def main():
    inp = get_input()
    text = dataconverter2.string_to_token_ids(inp)
    #create a copy but move the first dataconverter2.data_max_tokens/2 tokens to the end of the list
    text2 = text[dataconverter2.data_max_tokens//2:] + text[:dataconverter2.data_max_tokens//2]
    text = dataconverter2.normalize_token_length(text)
    text2 = dataconverter2.normalize_token_length(text2)
    print("predicting")
    ad = trainer2.predict(text)
    ad2 = trainer2.predict(text2)
    print(f"Ad: {ad}")
    print(f"Ad2: {ad2}")
    #get the string of the token that is the most likely to be an ad in the text
    mostad = dataconverter2.token_ids_to_string(text[ad.index(max(ad))])
    mostad_post = dataconverter2.token_ids_to_string(text[ad.index(max(ad))+1])
    mostad_pre = dataconverter2.token_ids_to_string(text[ad.index(max(ad))-1])
    print(f"most likely ad: {mostad_pre} \r\n {mostad} \r\n {mostad_post} \r\n\n")
    
    #get the string of the token that is the second most likely to be an ad in the text
    secondmostad = dataconverter2.token_ids_to_string(text[ad.index(sorted(ad)[-3])])
    secondmostad_post = dataconverter2.token_ids_to_string(text[ad.index(sorted(ad)[-3])+1])
    secondmostad_pre = dataconverter2.token_ids_to_string(text[ad.index(sorted(ad)[-2])-1])
    print(f"second most likely ad: {secondmostad_pre} \r\n {secondmostad} \r\n {secondmostad_post} \r\n\n")
    
    #get the string of the token that is the most likely to not be an ad in the text
    leastad = dataconverter2.token_ids_to_string(text[ad.index(min(ad))])
    leastad_post = dataconverter2.token_ids_to_string(text[ad.index(min(ad))+1])
    leastad_pre = dataconverter2.token_ids_to_string(text[ad.index(min(ad))-1])
    print(f"least likely ad: {leastad_pre} \r\n {leastad} \r\n {leastad_post} \r\n\n")
    
    #number of token lists to compare
    n = len(text)
    for i in range(n):
        
        # make sure the token is not mostly padding as this will mess up the evaluation
        num_zero = 0
        for token in text[i]:
            if token == 0:
                num_zero += 1
        
        if ad[i] > threshold and ad2[i] > threshold and num_zero < dataconverter2.data_max_tokens//2:
            #convert token ids to text
            str = dataconverter2.token_ids_to_string(text[i])
            # print the text in red
            print(f"\033[91m{str}\033[00m")
        else:
            #convert token ids to text
            str = dataconverter2.token_ids_to_string(text[i])
            # print the text in white
            print(f"\033[00m{str}\033[00m")    
    
if __name__ == "__main__":
    main()