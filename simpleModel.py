import numpy as np

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, AutoFeatureExtractor

def getSimpleModel(train_dataset, eval_dataset, metric, label2id, id2label, batch_size=32, model_checkpoint="facebook/wav2vec2-base"):

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    label2id, id2label = dict(), dict()

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-ks",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    return trainer
