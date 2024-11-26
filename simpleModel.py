import numpy as np

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, AutoFeatureExtractor

def getSimpleModel(train_dataset, eval_dataset, metric, label2id, id2label, batch_size=8, model_checkpoint="facebook/wav2vec2-base",gradient_accumulation_steps = 1, num_train_epochs = 10):

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-gtzan",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
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
