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
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, WordPunctTokenizer, TweetTokenizer, MWETokenizer, WhitespaceTokenizer, RegexpTokenizer


transcriptions_path = "./build/transcripts-sample.json"
tokens_path = "./build/tokens.csv"
output_path = "./build/train.csv"

#json example
#{"video_id": "fBxtS9BpVWs", "transcript": [{"text": "I need something to do a video about today. Oh, that won't be it", "start": 0.0, "duration": 4.89, "sponsor": false, "category": null, "action": null, "votes": null, "incorrectVotes": null, "reputation": null}, {"text": "No", "start": 7.96, "duration": 1.41, "sponsor": false, "category": null, "action": null, "votes": null, "incorrectVotes": null, "reputation": null}, 
def read_transcript_from_file():
    file = open(transcriptions_path, 'r')
    transcriptions = []    
    for line in file:
        transcript = json.loads(line)
        transcriptions.append(transcript)
    return transcriptions


def tokenize_transcriptions(transcriptions):
    tokenizer = TreebankWordTokenizer()
    tokens = []
    text = ""
    for transcript in transcriptions:
        for line in transcript['transcript']:
            temp = line['text']
            temp = temp.replace('\n', ' ')
            temp = temp.replace('\r', ' ')
            temp = temp.replace('\t', ' ')
            while '  ' in temp:
                temp = temp.replace('  ', ' ')
            text += temp
    tokens = tokenizer.tokenize(text)
    # save tokens to file
    df = pd.DataFrame(tokens)
    df.to_csv(tokens_path, index=True, header=True)
    return tokens

def string_to_tokens(string):
    #load tokens from file and convert string to tokens using the same tokenizer
    df = pd.read_csv(tokens_path)
    tokens = df['0'].tolist()
    return tokens

def tokens_to_string(tokens):
    #load tokens from file and convert tokens to string using the same tokenizer
    df = pd.read_csv(tokens_path)
    string = ""
    for token in tokens:
        
        string += token
def main():
    transcriptions = read_transcript_from_file()
    tokens = tokenize_transcriptions(transcriptions)
    string = "We featured at the link in the video description also down"
    tok = string_to_tokens(string)
    print(tok)
    str = tokens_to_string(tok)
    print(str)
    
if __name__ == "__main__":
    main()