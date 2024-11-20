import os
import glob
import json
import pandas as pd
import numpy as np

import torchaudio
import torch
import librosa
import IPython.display as ipd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoConfig, Wav2Vec2Processor, EvalPrediction, TrainingArguments

from trainer import DataCollatorCTCWithPadding
from model import Wav2Vec2ForSpeechClassification
from training import CTCTrainer

from transformers import (
    is_apex_available,
)

from packaging import version

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

DRUNK_THRESHOLD = 0.0005

is_regression = False

# model_name="lighteternal/wav2vec2-large-xlsr-53-greek"
model_name="maxidl/wav2vec2-large-xlsr-german"
pooling_mode="mean"

def load_data(debug = False):

    print("Starting to load data")
    data = []

    def find_bac(arr):
        for x in arr:
            if(x["name"] == "bak"):
                return float(x["value"])
        return 0

    dirs = os.listdir('./ALC')
    for i in dirs:
        json_files = glob.glob("./ALC/" + i + "/*.json")
        for file in json_files:
            with open(file, mode="r", encoding="cp1252") as read_file:
                json_data = json.loads(read_file.read())

            if len(json_data["levels"][0]["items"]) != 0:
                name = json_data["name"]
                file = os.path.dirname(file) + "/" + json_data["annotates"]
                bak = find_bac(json_data["levels"][0]["items"][0]["labels"])

                status = 'sober'
                if bak > DRUNK_THRESHOLD:
                    status = 'drunk'

                data.append([name, file, status])

    df = pd.DataFrame(np.array(data), columns=["name", "path", "drunken"])

    print("Data loaded")
    
    if debug:
        print(len(df), "data values")
        print("Labels ", df["drunken"].unique())
        print(df.groupby("drunken").count()[["path"]])
        print(df.head())

    return df

def listen(df):
    idx = np.random.randint(0, len(df))
    sample = df.iloc[idx]
    path = sample["path"]
    label = sample["drunken"]


    print(f"ID Location: {idx}")
    print(f"      Label: {label}")
    print()

    speech, sr = torchaudio.load(path)
    speech = speech[0].numpy().squeeze()
    speech = librosa.resample(np.asarray(speech), sr, 16_000)
    ipd.Audio(data=np.asarray(speech), autoplay=True, rate=16000)

def split_data(df, test_size=0.2):

    print("Creting train and test datasets")
    save_path = "./content/data"

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=101, stratify=df["drunken"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

    print("Train and test set created")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

def load_datasets():

    print("Loading datasets")

    data_files = {
        "train": "./content/data/train.csv", 
        "validation": "./content/data/test.csv",
    }

    # Entire dataset
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Partial dataset
    # train_dataset = load_dataset("csv", data_files=data_files, delimiter="\t", split="train[:1%]")
    # eval_dataset = load_dataset("csv", data_files=data_files, delimiter="\t", split="validation[:1%]")

    print("Datasets loaded")

    # print(train_dataset)
    # print(eval_dataset)

    return (train_dataset, eval_dataset)

def preprocess_datasets():

    (train_data, eval_data) = load_datasets()

    print("Starting preprocessing the data")
    
    input_column = "path"
    output_column = "drunken"

    label_list = train_data.unique(output_column)
    label_list.sort() 
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)

    processor = Wav2Vec2Processor.from_pretrained(model_name,)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    # print(f"The target sampling rate: {target_sampling_rate}")

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def label_to_id(label, label_list):

        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1

        return label

    def preprocess_function(examples):
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
        target_list = [label_to_id(label, label_list) for label in examples[output_column]]

        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result
    
    train_dataset = train_data.map(
        preprocess_function,
        batch_size=5,
        batched=True,
    )
    eval_dataset = eval_data.map(
        preprocess_function,
        batch_size=5,
        batched=True,
    )

    idx = 0
    # print(f"Training input_values: {train_dataset[idx]['input_values']}")
    # print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
    # print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['drunken']}")

    return (train_dataset, eval_dataset, processor, config)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

if __name__ == "__main__":
    
    df = load_data(True)
    # listen(df)
    split_data(df)

    (train_dataset, eval_dataset, processor, config) = preprocess_datasets()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True) 
        
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        output_dir="/content/wav2vec2-speech-drunkness-recognition",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=1.0,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=2,
    )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor,
    )

    trainer.train()