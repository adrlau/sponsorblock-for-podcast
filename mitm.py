from mitmproxy import ctx
import re
import os
import pyaudio
import wave
import whisper
import dataconverter2
import evaluater
model = whisper.load_model("base")

def transcribe_audio(filename):
    result = model.transcribe(filename)
    return result

def filter_ads(audiofile):
    transcript = transcribe_audio(audiofile)
    ctx.log.warn( "Transcript: {}".format( transcript.__dir__ ) )
    #tokenize the transcript
    return "test-audio.wav"
    
    

#TODO: rewrite this to not be blatant copy pasta from https://www.cron.dk/grabbing-media-with-mitmproxy/
class CybraryGrabber:
    def __init__(self):
        ctx.log.warn( "CybraryGrabber class initiated" )
        self.processed_file = None

    def writefile(self, filename, content):
        ctx.log.warn( "Writing File: {}".format( filename ) )
        # print( "Writing File: {}".format( filename ) )
        with open( filename, "wb" ) as f:
            f.write( content )
        f.close()

    def response(self, flow):
        url = flow.request.path
        #regex to check if the url is a audio file url. mp3, wav, ogg, etc. or contains the word audio
        is_audio = re.search( '.mp3|.wav|.ogg', url )
        if is_audio:
            ctx.log.warn( "Audio: {}".format( url ) )
            #get audop format
            audio_format = url.split('.')[-1]
            #make sure the audio format contains a valid file extension
            if "mp3" in audio_format.lower():
                audio_format = 'mp3'
            elif "wav" in audio_format.lower():
                audio_format = 'wav'
            elif "ogg" in audio_format.lower():
                audio_format = 'ogg'
            else:
                ctx.log.warn( "Audio format not supported: {}".format( audio_format ) )
                audio_format = ''
                return
                
                
            #hash the url to get a unique filename
            filename = str( hash( url ) )
            
            #create .tmp folder if it does not exist
            if not os.path.exists( ".tmp" ):
                os.makedirs( ".tmp" )
            self.writefile( ".tmp/" + filename + "." + audio_format, flow.response.content )
            
            #make sure the file is a valid audio file using pyaudio
            try:
                p = pyaudio.PyAudio()
                wf = wave.open( ".tmp/" + filename + "." + audio_format, 'rb' )
                #open the audio file in pyaudio
                stream = p.open( format=p.get_format_from_width( wf.getsampwidth() ),
                                 channels=wf.getnchannels(),
                                 rate=wf.getframerate(),
                                 output=True )
                
                
            except Exception as e:
                ctx.log.warn( "Audio file could not be played, exception: {}".format( e ) )
                #remove the file
                os.remove( ".tmp/" + filename + "." + audio_format )
                # return
            
            # filter_ads( ".tmp/" + filename + "." + audio_format )
            filter_ads( ".tmp/" + filename + "." + audio_format )
            processed_content = open( ".tmp/" + filename + "." + audio_format, 'rb' ).read()
            flow.response.content = processed_content
            
            #cleanup
            # os.remove( ".tmp/" + filename + "." + audio_format )
    
        

addons = [
    CybraryGrabber()
]

