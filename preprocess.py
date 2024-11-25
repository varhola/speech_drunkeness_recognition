import torchaudio
import evaluate

from datasets import load_dataset

from transformers import AutoConfig, Wav2Vec2Processor

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

    print(train_dataset)
    print(eval_dataset)

    return (train_dataset, eval_dataset)

def preprocess_datasets(model_name="maxidl/wav2vec2-large-xlsr-german", pooling_mode="mean"):

    (train_data, eval_data) = load_datasets()

    print("Starting preprocessing the data")
    
    input_column = "path"
    output_column = "drunken"

    label_list = train_data.unique(output_column)
    label_list.sort() 
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    label2id = {label: i for i, label in enumerate(label_list)},
    id2label = {i: label for i, label in enumerate(label_list)},

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
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

    metric = evaluate.load("accuracy")

    # idx = 0
    # print(f"Training input_values: {train_dataset[idx]['input_values']}")
    # print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
    # print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['drunken']}")

    return (train_dataset, eval_dataset, metric, label2id, id2label)
