import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np

imdb = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_train = imdb['train'].map(preprocess_function, batched=True)
tokenized_test = imdb['test'].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

training_args = TrainingArguments(
        output_dir='finetune-sentiment-model',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch", 
        push_to_hub=False,
        )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate on test data
trainer.evaluate()
