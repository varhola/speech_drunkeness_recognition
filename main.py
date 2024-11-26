from huggingface_hub import notebook_login

from simpleModel import getSimpleModel
from preprocess import preprocess_datasets
from readData import create_database

if __name__ == "__main__":

    notebook_login()

    create_database()

    (train_dataset, eval_dataset, metric, label2id, id2label, id2label_fn) = preprocess_datasets()

    trainer = getSimpleModel(train_dataset, eval_dataset, metric, label2id, id2label)
    trainer.train()
    trainer.evaluate()
