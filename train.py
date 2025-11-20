import os
import datasets
import soundfile as sf
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from jiwer import wer
import librosa

dataset_path = r"C:\Users\qurtz\Desktop\projects\audioproject\dataset"

def load_split(subfolder):
    items = []
    folder = os.path.join(dataset_path, subfolder)
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            wav = os.path.join(folder, f)
            txt = os.path.join(folder, f.replace(".wav", ".txt"))
            if os.path.exists(txt):
                items.append({
                    "audio": wav,
                    "text": open(txt, encoding="utf-8").read().strip()
                })
    return items

train_items = load_split("train")
test_items = load_split("testt")

train_dataset = datasets.Dataset.from_list(train_items)
test_dataset = datasets.Dataset.from_list(test_items)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Kazakh",
    task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


def load_audio(example):
    audio, sr = librosa.load(example["audio"], sr=16000, mono=True)
    example["audio_array"] = audio
    example["sampling_rate"] = sr
    return example

train_dataset = train_dataset.map(load_audio)
test_dataset = test_dataset.map(load_audio)


def prepare(example):
    inputs = processor(
        example["audio_array"],
        sampling_rate=example["sampling_rate"],
        return_tensors="pt"
    )

    example["input_features"] = inputs.input_features[0]

    labels = processor.tokenizer(
        example["text"],
        return_tensors="pt",
    ).input_ids[0]

    example["labels"] = labels
    return example

train_dataset = train_dataset.map(prepare)
test_dataset = test_dataset.map(prepare)

train_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

class MyWhisperCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, batch):
        # Pad input_features
        input_features = [b["input_features"] for b in batch]
        input_features = torch.nn.utils.rnn.pad_sequence(
            input_features,
            batch_first=True,
            padding_value=0.0
        )

        # Pad labels
        labels = [b["labels"] for b in batch]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_features": input_features,
            "labels": labels,
        }

data_collator = MyWhisperCollator(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.bos_token_id
)

args = Seq2SeqTrainingArguments(
    output_dir="whisper_kk_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=10,
    fp16=False,  # CPU only
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.train()


def eval_wer(dataset):
    refs, hyps = [], []
    for ex in dataset:
        feats = processor(
            ex["audio_array"],
            sampling_rate=ex["sampling_rate"],
            return_tensors="pt"
        ).input_features

        pred = model.generate(feats)
        text = processor.tokenizer.decode(pred[0], skip_special_tokens=True)

        refs.append(ex["text"].lower())
        hyps.append(text.lower())

    return wer(refs, hyps)

print("Final WER:", eval_wer(test_dataset))
