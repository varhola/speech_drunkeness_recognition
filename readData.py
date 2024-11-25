import os
import glob
import json
import torchaudio
import librosa
import pandas as pd
import numpy as np
import IPython.display as ipd

from sklearn.model_selection import train_test_split

DRUNK_THRESHOLD = 0.0005

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

def create_database():
    df = load_data(True)
    # listen(df)
    split_data(df)