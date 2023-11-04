import whisper

model = whisper.load_model("base")
# result = model.transcribe("test-audio.wav")
result = model.transcribe("dd137.mp3")
print(result["text"])