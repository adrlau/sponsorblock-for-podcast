from pydub import AudioSegment
import os

# load test audio
audio = AudioSegment.from_file("dd.mp3", format="mp3")
#convert to wav
audio.export("dd.wav", format="wav")
#import wav file
audio = AudioSegment.from_file("dd.wav", format="wav")

#remove part from audio
start = 0
end = 500
audio = audio[:int(start*1000)] + audio[int(end*1000):]
audio.export("dd2.wav", format="wav")