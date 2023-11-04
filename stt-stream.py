import whisper
import pyaudio
import wave

# Load the Whisper model
model = whisper.load_model("base")

# Define audio stream parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "dd137.mp3"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Start the audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

# Record audio in chunks and append to frames
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Transcribe the audio file
result = model.transcribe(WAVE_OUTPUT_FILENAME)

# Print the transcription
print(result["text"])