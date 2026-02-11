import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


id2label = {
    0: "体育", 1: "财经", 2: "房产", 3: "家居", 4: "教育",
    5: "科技", 6: "时尚", 7: "时政", 8: "游戏", 9: "娱乐"
}
num_labels = len(id2label)
print(f"标签映射表: {id2label}")


print("正在读取数据...")


train_df = pd.read_csv("data/cnews.train.txt", sep="	", names=["text", "label"]).head(2000)
test_df = pd.read_csv("data/cnews.test.txt", sep="	", names=["text", "label"]).head(500)


train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


model_name = "bert-base-chinese"
print(f"正在加载分词器: {model_name} ...")
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("正在对文本进行分词...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)


tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")


print(f"正在下载/加载模型: {model_name} ...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir="./cnews_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)


print("=== 开始微调训练 ===")
trainer.train()

