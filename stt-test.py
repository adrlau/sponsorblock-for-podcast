import whisper

model = whisper.load_model("base")
# result = model.transcribe("test-audio.wav")
result = model.transcribe("wan.mp3")
print(result["text"])