import whisper

model = whisper.load_model("base")
result = model.transcribe("stt-test.mp3")
print(result["text"])