"""
This file is intended to be used to generate the dataset to train our AI model for ad/sponsor detection on.
The data about witch videos are used and where the ads/sponshorships are is taken from sponsorblock for youtube (https://sponsor.ajay.app/).


Transcript sample:
{'text': "Antarctica is Earth's coolest continent,", 'start': 0.0, 'duration': 2.52}, {'text': 'and most complicatedly claimed continent.', 'start': 2.52, 'duration': 2.1},

Sponsor data from sponsorblock
videoID,startTime,endTime,votes,locked,incorrectVotes,UUID,userID,timeSubmitted,views,category,actionType,service,videoDuration,hidden,reputation,shadowHidden,hashedVideoID,userAgent,description
U0wTDK0VOeY,148.429,164.348,2,1,1,3a7298b2e5b21d37c06581037feed944336e5780551ea0ed399d2247b56854da7,6b1c73ae7b1e60aef2d73c5de8f3ee06d220f836e0f8c5daa9255f8c2b4a2c9b,1680007430870,548,filler,skip,YouTube,299.001,0,27,0,93646c719490256e8cb43cfaa41e39534525389b0b28f5d8f28ca937f1abcf9d,sponsorBlocker@ajay.app/v5.3.1,""

need to first get all the transcripts for all the videos in the dataset.
we need to filter out a few things:
    - videos that are not in english
then we need to get all the timestamps for the ads/sponsorships of videos we have transcripts for.
"""

from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import json


sponsordata_path = "./sb-mirror/sponsorTimes-sample.csv"
videos_path = "./youtube_videos.json"
transcriptions_path = "./transcripts.json"

time_threshold = 1288466965 # 2010 in unix time (seconds) to awoid very old videos
video_duration_threshold = 61 # 1 minute in seconds to awoid yt shorts. as most podcasts are longer than 1 minute and yt shorts have a very different format than podcasts.
ad_duration_threshold = 5 # to filter out ads that are too short to be useful
views_threshold = 20 # to filter out sponsors with very few views. Reccomend to keep this low.
reputation_threshold = 5
votes_threshold = 10 # to filter out sponsorships with very few votes. Reccomend to keep this low.
votes_ratio_threshold = 1.5 # to filter out sponsorships with a lot of downvotes as they are probably mislabeled. 0 means equal upvotes and downvotes, 1 means only upvotes, -1 means only downvotes

def get_all_youtube_video_ids():
    
    #debug
    print("loading sponsorTimes")
    
    df = pd.read_csv(sponsordata_path, low_memory=False)
    
    #debug
    print("loaded sponsorTimes")
    print(f"number of rows: {len(df)}")
    print("Filtering out hidden and non yt videos")
    
    # filter out
    # remove all the videos that are not from youtube
    df = df[df['service'] == 'YouTube']
    
    #remove all videos with are hidden or shadowHidden
    df = df[df['hidden'] == 0]
    df = df[df['shadowHidden'] == 0]
    
    #debug
    print("Filtering custom parameters")
    print(f"number of rows1: {len(df)}")
    
    # filter out based on custom parameters
    
    
    # remove all the videos that are shorter than duration_threshold seconds if the dureation is set to something other than 0
    df = df.drop(df[(df['videoDuration'] != False) & (df['videoDuration'] < video_duration_threshold)].index)
    
    print(f"number of rows after video durationfilter: {len(df)}")
    
    # remove all the videos that are shorter than ad_duration_threshold seconds from the duration between startTime and endTime
    df = df.drop(df[(df['endTime'] - df['startTime'] < ad_duration_threshold)].index)
    
    print(f"number of rows after ad durationfilter: {len(df)}")
    
    # remove all the videos that have less than views_threshold views
    df = df.drop(df[(df['views'] != False) & (df['views'] < views_threshold)].index)
    
    print(f"number of rows after views filter: {len(df)}")
    
    # remove all the videos that have less than reputation_threshold reputation
    df = df.drop(df[(df['reputation'] != False) & (df['reputation'] < reputation_threshold)].index)
    
    print(f"number of rows after reputation filter: {len(df)}")
    
    # remove all the videos that have less votes than votes_threshold
    df = df.drop(df[df['votes'] < votes_threshold].index)
    print(f"number of rows after votes filter: {len(df)}")
    
    # remove all the videos that have a negative votes ratio
    df = df.drop(df[(df['votes'] - df['incorrectVotes']) < votes_ratio_threshold].index)
    
    
    print(f"number of rows after votes ratio filter: {len(df)}")
    
    #debug
    print("Parsing videos and timestamps")
    print(f"number of rows: {len(df)}")
    
    # get all categories
    categories = df['category'].unique()
    
    # get all actions
    actions = df['actionType'].unique()
    
    # get a list of all unique video ids
    youtube_video_ids = [video_id for video_id in df['videoID'].unique() if video_id != '']
    
    #get a dict of all the video ids, timestamps with tuples of start and end times, categories and actions for the timestamp for each video
    youtube_videos = {}
    for video_id in youtube_video_ids:
        
        # print progress
        # print("processing video: " + video_id)
        
        video = { 'video_id': video_id , 'timestamps': []}
        # get all entries in the df for the video
        df_video = df[df['videoID'] == video_id]
        for index, row in df_video.iterrows():
            video['timestamps'].append(({ 'start': row['startTime'], 'end': row['endTime'], 'category': row['category'], 'action': row['actionType'], 'votes': row['votes'], 'incorrectVotes': row['incorrectVotes'], 'reputation': row['reputation']}))
        
        # debug
        # print("video processed: ")
        # print(video)
        
        # add the video to the dict
        youtube_videos[video_id] = video
        
    
    return youtube_videos, youtube_video_ids, categories, actions

def write_youtube_videos_to_file(youtube_videos):
    with open(videos_path, 'w') as outfile:
        json.dump(youtube_videos, outfile)

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return transcript
    except:
        return None
    
def add_sponsor_tags_to_transcript(transcript, timestamps):
    for i in transcript:
        for timestamp in timestamps:
            if i['start'] >= timestamp['start'] and i['start'] <= timestamp['end']:
                i['sponsor'] = True
                i['category'] = timestamp['category']
                i['action'] = timestamp['action']
                i['votes'] = timestamp['votes']
                i['incorrectVotes'] = timestamp['incorrectVotes']
                i['reputation'] = timestamp['reputation']
                break
        else:
            i['sponsor'] = False
            i['category'] = None
            i['action'] = None
            i['votes'] = None
            i['incorrectVotes'] = None
            i['reputation'] = None
    return transcript

def save_transcript_to_file(transcript, video_id):
    # save the transcript to a file
    data = {"video_id": video_id, "transcript": transcript}
    #append to file
    with open(transcriptions_path, 'a') as outfile:
        json.dump(data, outfile)
        outfile.write("\n")

def get_transcript_from_file_by_id(video_id):
    file = open(transcriptions_path, 'r')
    for line in file:
        data = json.loads(line)
        if data['video_id'] == video_id:
            return data['transcript']
        else:
            # skip to next line
            continue
    return None

def read_transcript_from_file(index):
    file = open(transcriptions_path, 'r')
    for i, line in enumerate(file):
        if i == index:
            data = json.loads(line)
            return data['transcript']
        else:
            # skip to next line
            continue
    return None

def display_transcript(transcript):
    # display the transcript with sponsored parts in red and non sponsored parts in white. If the display does not support colors it will just display the transcript without colors but with __ around sponsored parts
    try:
        import colorama
        colorama.init()
        for i in transcript:
            if i['sponsor']:
                print(colorama.Fore.RED + i['text'] + colorama.Style.RESET_ALL)
            else:
                print(i['text'])
    except:
        for i in transcript:
            if i['sponsor']:
                print("__" + i['text'] + "__")
            else:
                print(i['text'])



def main():
    print("Getting youtube_videos")
    youtube_videos = get_all_youtube_video_ids()
    print("Saving youtube video data to file")
    write_youtube_videos_to_file(youtube_videos[0])
    print("Getting transcript for each video")
    video_ids = youtube_videos[1]
    for id in video_ids:
        # check if we already have the transcript for this video
        existing_transcript = get_transcript_from_file_by_id(id)
        if  existing_transcript != None:
            continue
        transcript = get_transcript(id)
        if transcript == None:
            continue
        transcript = add_sponsor_tags_to_transcript(transcript, youtube_videos[0][id]['timestamps'])
        print("Saving transcript for video: " + id)
        save_transcript_to_file(transcript, id)
    
    transcript = read_transcript_from_file(2)
    display_transcript(transcript)

if __name__ == "__main__":
    main()