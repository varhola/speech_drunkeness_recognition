import torchaudio
import evaluate

from datasets import load_dataset

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def load_datasets():

    print("Loading datasets")

    data_files = {
        "train": "./content/data/train.csv", 
        "validation": "./content/data/test.csv",
    }

    # Entire dataset
    # dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["validation"]

    # Partial dataset
    train_dataset = load_dataset("csv", data_files=data_files, delimiter="\t", split="train[:1%]")
    eval_dataset = load_dataset("csv", data_files=data_files, delimiter="\t", split="validation[:1%]")

    print("Datasets loaded")

    return (train_dataset, eval_dataset)

def id2label_fn(id):
    labels = ["sober", "drunk"]
    return labels[id]

def preprocess_datasets():

    (train_data, eval_data) = load_datasets()

    print("Starting preprocessing the data")
    
    input_column = "path"
    output_column = "drunken"

    label_list = train_data.unique(output_column)
    label_list.sort()
    num_labels = len(label_list)

    label2id = {
        "sober": 0,
        "drunk": 1,
    },
    id2label = {
        0: "sober",
        1: "drunk",
    }

    model_id = "ntu-spml/distilhubert"
    
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    print(f"A classification problem with {num_labels} classes")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )
    target_sampling_rate = feature_extractor.sampling_rate
    print("Sampling rate:", target_sampling_rate)

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def preprocess_function(examples):
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
        target_list = [label for label in examples[output_column]]

        max_duration = 5.0

        inputs = feature_extractor(
            speech_list, 
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )

        inputs["label"] = list(target_list)

        return inputs
    
    train_dataset = train_data.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        remove_columns=["drunken", "name", "path"],
    )
    eval_dataset = eval_data.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        remove_columns=["drunken", "name", "path"],
    )

    return (train_dataset, eval_dataset, model, feature_extractor)
