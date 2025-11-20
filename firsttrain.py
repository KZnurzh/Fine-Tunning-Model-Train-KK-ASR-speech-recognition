from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr(r"C:\Users\qurtz\Desktop\projects\audioproject\dataset\train\audio1.wav")
print(result["text"])
