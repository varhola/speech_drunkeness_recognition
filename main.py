from huggingface_hub import login

from simpleModel import getSimpleModel
from preprocess import preprocess_datasets
from readData import create_database

if __name__ == "__main__":

    login()

    print("Starting a new run")

    # create_database()

    (train_dataset, eval_dataset, model, feature_extractor) = preprocess_datasets()

    trainer = getSimpleModel(train_dataset, eval_dataset, model, feature_extractor)
    trainer.train()
    trainer.evaluate()

    print("Run finished")
