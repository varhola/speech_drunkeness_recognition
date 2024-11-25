from trainer import DataCollatorCTCWithPadding
from model import Wav2Vec2ForSpeechClassification
from training import CTCTrainer

from transformers import EvalPrediction, TrainingArguments

import numpy as np

is_regression = False

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def get_complex_model(model_name, processor, config, train_dataset, eval_dataset):
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

    return trainer