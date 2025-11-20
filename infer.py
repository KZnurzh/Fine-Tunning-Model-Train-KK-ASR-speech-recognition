import soundfile as sf
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

model_name = r"C:\Users\qurtz\Desktop\projects\audioproject\whisper_kk_model"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

audio_path = r"C:\Users\qurtz\Desktop\projects\audioproject\dataset\train\audio1.wav"

speech, sr = sf.read(audio_path)

if sr != 16000:
    speech = torchaudio.functional.resample(
        torch.from_numpy(speech).float(), sr, 16000
    ).numpy()
    sr = 16000

input_values = processor(
    speech,
    sampling_rate=sr,
    return_tensors="pt"
).input_values

logits = model(input_values).logits
pred_ids = logits.argmax(dim=-1)
text = processor.batch_decode(pred_ids)[0]

print("TRANSCRIPT:", text)
