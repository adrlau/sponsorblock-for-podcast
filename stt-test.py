import whisper

model = whisper.load_model("base")
result = model.transcribe("test-audio.wav")
print(result["text"])