# this script converts the json file of transcriptions into a csv file with trainable data for the model. 
# the csv file will be in the following format:
# is_ad, ad_start_char, transcript
# 0, 0, "This is a transcript"
# 1, 0, "This is an ad"
# 1, 24, "look at this cool thing This is another ad"
# is_ad is a 0 or 1 indicating whether the transcript contains an ad or not
# the ad_start_char is the index of the first character of the ad in the transcript 
# the transcript is the transcript of the video

import pandas as pd
import json
import os
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

from collections import Counter
from itertools import repeat, chain
import string 
import multiprocessing
import concurrent.futures

transcriptions_path = "./build/transcripts-sample.json"
tokens_path = "./build/tokens.csv"
data_path = "./build/data.csv"
ad_keywords_path = "./blocklist.txt" #added to force in a blocklist with specially bad or triggering words in the dataset.
data_max_tokens = 16
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

#json example
#{"video_id": "fBxtS9BpVWs", "transcript": [{"text": "I need something to do a video about today. Oh, that won't be it", "start": 0.0, "duration": 4.89, "sponsor": false, "category": null, "action": null, "votes": null, "incorrectVotes": null, "reputation": null}, {"text": "No", "start": 7.96, "duration": 1.41, "sponsor": false, "category": null, "action": null, "votes": null, "incorrectVotes": null, "reputation": null}, 
def read_transcript_from_file():
    file = open(transcriptions_path, 'r')
    transcriptions = []    
    for line in file:
        try:
            transcript = json.loads(line)
            transcriptions.append(transcript)
        except:
            continue
    return transcriptions

def stringparser(text, debug=False):
    #lemmatize text to reduce the number of unique tokens in the dataset
    # Convert text to lowercase
    text = text.lower()
    # Replace unwanted characters with appropriate substitutes
    replacements = {
        '"': '',
        "'": '',
        '`': '',
        '´': '',
        '.': ' . ',
        '-': ' ',
        '_': ' ',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
        "": '',
        "": '',
        "  ": ' '
    }
    if debug:
        print(f"Starting to Lemmatize")
    #lemmatize text to reduce language complexity
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    if debug:
        print(f"Done Lemmatizing starting to replace characters")
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove non-printable characters
    text = ''.join(char for char in text if char in string.printable)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def create_token_tree_from_transcriptions(debug=False):
    transcriptions = read_transcript_from_file()
    tokens = []
    text = ""
    for transcript in transcriptions:
        if transcriptions.index(transcript) % 100 == 0:
            if debug:
                print(f"parsing transcript nr {transcriptions.index(transcript)}")
        for line in transcript['transcript']:
            temp = stringparser(line['text'])
            text += " "+ temp + " "
    if debug:
        print("tokenizing text")
    tokens = tokenizer.tokenize(text)
    # add each char from printable asci to the end of the list to make sure everything is tokenizable
    if debug:
        print("adding single characters to tokens")
    for char in string.printable:
        if char not in ['\n', '\r', '\t']:
            tokens.append(char)
    #sort list based on number of occurences of each token from most to least
    tokens = list(chain.from_iterable(repeat(i, c) for i,c in Counter(tokens).most_common()))
    # convert tokens to dataframe and save to file. but keep only unique tokens in the dataframe
    df = pd.DataFrame(tokens).drop_duplicates()
    df = df.reset_index(drop=True)
    df.to_csv(tokens_path, index=True, header=False)
    # write ower first line to tokens file with the proper header id,text
    with open(tokens_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("id,text\n" + content)
    return tokens

def string_to_token_ids(string, debug=False):
    token_dataframe = pd.read_csv(tokens_path)
    string = stringparser(string, debug=debug)
    if debug:
        print(f"Done parsing")
    tokens = tokenizer.tokenize(string)
    #convert tokens to list of ids in df using the tokens in the df as a reference. If a token is not in the df, add it to the df convert it to single letter token and add it to the list of ids
    ids = []
    for token in tokens:
        #find id of the text in the df
        id = token_dataframe[token_dataframe['text'] == token].index.values
        if debug:
            print(f"token: {token} id: {id}")
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

def token_ids_to_string(ids):
    df = pd.read_csv(tokens_path)
    string = ""
    for id in ids:
        string += df['text'][id] + " "
    return string

def normalize_token_length(tokens):
    tokens = [tokens[i:i + data_max_tokens] for i in range(0, len(tokens), data_max_tokens)]
    #pad the last list of tokens with 0s to make it the same length as the rest
    for i in range(len(tokens[-1]), data_max_tokens):
        tokens[-1].append(0)
    return tokens

def generate_dataset(debug=False):
    transcriptions = read_transcript_from_file()
    df1 = pd.read_csv(tokens_path)
    file = open(data_path, 'w')
    file.write("is_ad,token_ids\n")
    file.close()
    
    ad_string = ""
    no_ad_string = ""
    for char in string.printable:
        if char not in ['\n', '\r', '\t']:
            no_ad_string += char
            ad_string += char
    
    ad_string += ". "
    no_ad_string += ". "
    
    #read blocklist and add all words to the ad_string
    blocklist = open(ad_keywords_path, 'r')
    for line in blocklist:
        ad_string += line + ". "
    
    
    for transcript in transcriptions:
        if debug:
            print(f"generating dataset from transcript {transcriptions.index(transcript)}")
        script = transcript['transcript']
        for sentence in script:
            if sentence['sponsor']:
                ad_string += sentence['text'] + " "
            else:
                no_ad_string += sentence['text'] + " "
    if debug:
        print("generating token ids ads")
    ad_tokens = string_to_token_ids(ad_string, debug=debug)
    if debug:
        print("generating token ids non ads")
    no_ad_tokens = string_to_token_ids(no_ad_string, debug=debug)
    if debug:
        print("Prosessing")
    # split the tokens into lists of tokens with max length of data_max_tokens
    ad_tokens = normalize_token_length(ad_tokens)
    no_ad_tokens = normalize_token_length(no_ad_tokens)
    
    if debug:
        print("Writing to file")
    
    for token_ids in ad_tokens:
        df = pd.DataFrame([{'is_ad': 1, 'token_ids': token_ids}])
        df.to_csv(data_path, mode='a', index=False, header=False)
    for token_ids in no_ad_tokens:
        df = pd.DataFrame([{'is_ad': 0, 'token_ids': token_ids}])
        df.to_csv(data_path, mode='a', index=False, header=False)

def main():
    print("reading transcripts from file")
    print("tokenizing transcriptions")
    #check if tokens file exists
    tokens = create_token_tree_from_transcriptions(debug=False)
    print("tokens saved to file")
    string = "We featured at the link in the video description also down"
    print("testing tokenization on:" + string)
    
    tok = string_to_token_ids(string)
    print(tok)
    str = token_ids_to_string(tok)
    print(str)
    #test to tokenize the alphabet and convert it back to string
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!:;\"\' "
    print(f"trying to tokenize {alphabet}")
    tok = string_to_token_ids(alphabet)
    print(tok)
    alphabet = token_ids_to_string(tok)
    print(alphabet)
    print("generating dataset")
    generate_dataset(debug=False)
    print("done generating dataset")
    
if __name__ == "__main__":
    main()