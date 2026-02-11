import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# from intent.training_code.train_bert import train_labels

dataset_df = pd.read_csv("/data0/yhguo/badouAI/Week04/intent_dataset_12000.csv", sep=",", header=None)
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df[1].values[:500])
texts = list(dataset_df[0].values[:500])

x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained("/data0/yhguo/badouAI/intent/assets/models/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained('/data0/yhguo/badouAI/intent/assets/models/bert-base-chinese', num_labels=12)

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset=Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(model=model,
                  args = training_args,
                  train_dataset=train_dataset,
                  eval_dataset = test_dataset,
                  compute_metrics = compute_metrics,
                  )

trainer.train()
trainer.evaluate()
