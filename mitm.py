from mitmproxy import ctx
import re
import os
import pyaudio
import wave
import whisper
import dataconverter2
import evaluater
import pydub
from pydub import AudioSegment
import threading
import asyncio
from mitmproxy.script import concurrent


model = whisper.load_model("tiny.en")
class MITMGrabber:
    def __init__(self):
        ctx.log.warn( "Starting" )
        self.processed_file = None

    def transcribe_audio(self, filename):
        result = model.transcribe(filename)
        return result # result["text"] is plaintext transcript result["segments"] is a list of segments with start and end time and text

    def filter_ads(self, audiofile):
        # return ".tmp/test-audio.wav"
        audio_format = audiofile.split(".")[-1]
        transcript = self.transcribe_audio(audiofile)
        segments = transcript["segments"]
        text = transcript["text"]
        tokens = dataconverter2.string_to_token_ids(text)
        ad, normalized_tokens, estimates = evaluater.is_ad_tokens(tokens)
        ctx.log.warn("ad: {}".format(ad))
        #save transcript to file
        filename = audiofile + ".transcript.txt"
        file = open(filename, 'w')
        file.write(text)
        
        audio_time_length = AudioSegment.from_file(audiofile).duration_seconds
        segment_length = audio_time_length / len(ad)
        
        for i in range(len(ad)):
            a = ad[i]
            ctx.log.warn("ad at index {}: {}".format(i, a))
            if a:
                # remove part from audio
                start = i * segment_length
                end = start + segment_length
                
                ctx.log.warn("start: {}, end: {}".format(start, end))
                audio = AudioSegment.from_file(audiofile)
                audio = audio[:int(start*1000)] + audio[int(end*1000):]
                audio.export(audiofile, format=audio_format)
        
        
        output_file = audiofile + ".filtered." + audio_format + ".wav"
        audio = AudioSegment.from_file(audiofile)
        audio.export(output_file, format="wav")
        return output_file
        
    def writefile(self, filename, content):
        if not os.path.exists( ".tmp" ):
            os.makedirs( ".tmp" )
        with open(filename, "wb" ) as f:
            f.write( content )
        f.close()

    async def response(self, flow):
        # def process_response(self):
            url = flow.request.path
            
            content_type = flow.response.headers.get("Content-Type", "")
            
            is_audio = re.search('audio|mpeg|mp3|wav|ogg', content_type, re.IGNORECASE)
            if is_audio:
                ctx.log.warn("Audio: {}".format(url))
                audio_format = re.search('mpeg|mp3|wav|ogg', content_type, re.IGNORECASE).group(0)
                filename = str(hash(url))
                filename = ".tmp/" + filename + "." + audio_format
                self.writefile(filename, flow.response.content)
                
                ctx.log.warn("Audio file loaded filtering ads")
                try:
                    filename = self.filter_ads(filename)
                    processed_content = open(filename, 'rb').read()
                    flow.response.content = processed_content
                except Exception as e:
                    ctx.log.warn("Audio file could not be filtered, exception: {}".format(e))
                    return
        # thread = threading.Thread(target=process_response, args=(self,))
        # thread.start()

addons = [
    MITMGrabber()
]

