# this script converts the json file of transcriptions into a csv file with trainable data for the model. 
# the csv file will be in the following format:
# is_ad, ad_start_char, transcript
# 0, 0, "This is a transcript"
# 1, 0, "This is an ad"
# 1, 24, "look at this cool thing This is another ad"
# is_ad is a 0 or 1 indicating whether the transcript contains an ad or not
# the ad_start_char is the index of the first character of the ad in the transcript 
# the transcript is the transcript of the video

import json
import string
import pandas as pd
from multiprocessing import pool
import textacy
import textacy.preprocessing as tprep
from textacy import representations
import spacy

nlp = spacy.load("en_core_web_sm")

transcriptions_path = "./build/transcripts-sample.json"
data_path = "./build/data3.csv"
min_text_length = 16 #minimum length of a text in characters to be included in the dataset

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


def preprocess_text(text):
    
    #remove newlines and useless whitespace
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace(",", " ")
    
    #preprocess
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.unicode(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.remove.accents(text)
    text = tprep.replace.urls(text)
    text = tprep.replace.user_handles(text)
    text = tprep.remove.punctuation(text)
    text = tprep.remove.digits(text) #  untested
    text = tprep.remove.stopwords(text) #  untested
    text = tprep.normalize.whitespace(text)
    
    #remove double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    #lemmatize
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])
    
    return text
    
    
    
    
from multiprocessing import Pool

def process_transcript(transcript, debug=True):
    dataset = pd.DataFrame(columns=['is_ad', 'text'])

    current_part = 0
    current_text = ""
    last_part = 0
    for part in transcript["transcript"]:
        current_part = int(part["sponsor"])
        if current_part == last_part:
            current_text += " " + part["text"]
            continue
        else :
            #add the last part to the dataset
            text = current_text
            if text != "":
                #TODO: test if it is best to keep long texts or split them
                #split the text into multiple texts if it is too long
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                for sentence in sentences:
                    sentence = preprocess_text(sentence)
                    dataset = pd.concat([dataset, pd.DataFrame({'is_ad': [last_part], 'text': [sentence]})], ignore_index=True)
                # #still add the long text to the dataset to make sure the model can learn from it even if it is not perfect for evaluation
                # text = preprocess_text(text)
                # dataset = pd.concat([dataset, pd.DataFrame({'is_ad': [last_part], 'text': [text]})], ignore_index=True)
            current_text = part["text"]
        last_part = current_part
    return dataset

def postprocess_dataset(dataset):
    #TODO: test this function and include it in the dataset generation
    #remove all rows with empty text or text that is too short to be useful
    dataset = dataset[dataset["text"].str.len() > min_text_length]
    
    #remove all rows with duplicate text
    dataset = dataset.drop_duplicates(subset=['text'])
    
    # normalize amouiunt of ads and non ads
    num_ads = len(dataset[dataset["is_ad"] == 1])
    num_non_ads = len(dataset[dataset["is_ad"] == 0])
    #shuffle the dataset to remove random bias
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    if num_ads > num_non_ads:
        dataset = dataset.drop(dataset[dataset["is_ad"] == 1].sample(n=num_ads-num_non_ads).index)
    elif num_non_ads > num_ads:
        dataset = dataset.drop(dataset[dataset["is_ad"] == 0].sample(n=num_non_ads-num_ads).index)
    return dataset

def generate_dataset(debug=False, progress=False):
    transcripts = read_transcript_from_file()
    
    #generate the dataset
    dataset = pd.DataFrame(columns=['is_ad', 'text'])
    prog = 0 
    with Pool() as pool:
        for transcript_dataset in pool.imap_unordered(process_transcript, transcripts):
            if progress or debug:
                percent = int(prog/len(transcripts)*100)
                if percent % 1 == 0:
                    print(f"{percent}% complete  processing video {prog}/{len(transcripts)} dataset size: {len(dataset)}")
            prog += 1
            dataset = pd.concat([dataset, transcript_dataset], ignore_index=True)
    
    if debug:
        print("splitting long texts")
    #split excessively long texts into multiple texts with the same label.
    
    
    
    return dataset
                
                
def main():
    dataset = generate_dataset(debug=False, progress=True)
    # clear the file
    open(data_path, 'w').close()
    # save the dataset to file
    dataset.to_csv(data_path, index=False)
    print("done")
    
if __name__ == "__main__":
    main()