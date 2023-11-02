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
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, WordPunctTokenizer, TweetTokenizer, MWETokenizer, WhitespaceTokenizer, RegexpTokenizer

from collections import Counter
from itertools import repeat, chain
import string 

transcriptions_path = "./build/transcripts-sample.json"
tokens_path = "./build/tokens.csv"
data_path = "./build/data.csv"
data_max_tokens = 50
tokenizer = TreebankWordTokenizer()

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


def create_token_tree_from_transcriptions(transcriptions):
    tokens = []
    text = ""
    for transcript in transcriptions:
        if transcriptions.index(transcript) % 100 == 0:
            print(f"parsing transcript nr {transcriptions.index(transcript)}")
        for line in transcript['transcript']:
            temp = stringparser(line['text'])
            text += " "+ temp + " "
            
    print("tokenizing text")
    tokens = tokenizer.tokenize(text)
    # add each char from printable asci to the end of the list to make sure everything is tokenizable
    for char in string.printable:
        if char not in ['\n', '\r', '\t']:
            tokens.append(char)
    #sort list based on number of occurences of each token from most to least
    tokens = list(chain.from_iterable(repeat(i, c) for i,c in Counter(tokens).most_common()))
    # convert tokens to dataframe and save to file. but keep only unique tokens in the dataframe
    df = pd.DataFrame(tokens).drop_duplicates()
    df.reset_index(drop=True)
    df.to_csv(tokens_path, index=True, header=False)
    # write ower first line to tokens file with the proper header id,text
    with open(tokens_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("id,text\n" + content)
    return tokens

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

def token_ids_to_string(ids):
    df = pd.read_csv(tokens_path)
    string = ""
    for id in ids:
        string += df['text'][id] + " "
    return string


def generate_dataset(transcriptions, token_dataframe):
    df1 = pd.read_csv(tokens_path)
    first_write = True
    for transcript in transcriptions:
        print(f"generating dataset from transcript {transcript['video_id']}")
        for line in transcript['transcript']:
            text = line['text']
            token_ids = string_to_token_ids(text, token_dataframe)
            is_ad = 1 if line['sponsor'] else 0
            df = pd.DataFrame([{'is_ad': is_ad, 'token_ids': token_ids}])
            if first_write:
                df.to_csv(data_path, index=False, header=True)
                first_write = False
            else:
                df.to_csv(data_path, mode='a', index=False, header=False)

def main():
    print("rading transcripts from file")
    transcriptions = read_transcript_from_file()
    print("tokenizing transcriptions")
    #check if tokens file exists
    tokens = create_token_tree_from_transcriptions(transcriptions)
    
    
    print("tokens saved to file")
    string = "We featured at the link in the video description also down"
    print("testing tokenization on:" + string)
    
    token_dataframe = pd.read_csv(tokens_path)
    
    tok = string_to_token_ids(string, token_dataframe)
    print(tok)
    str = token_ids_to_string(tok)
    print(str)
    #test to tokenize the alphabet and convert it back to string
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!:;\"\' "
    print(f"trying to tokenize {alphabet}")
    tok = string_to_token_ids(alphabet, token_dataframe)
    print(tok)
    alphabet = token_ids_to_string(tok)
    print(alphabet)
    print("generating dataset")
    generate_dataset(transcriptions, token_dataframe)
    print("done generating dataset")
    
if __name__ == "__main__":
    main()