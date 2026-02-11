import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import numpy as np

#加载公开中文数据集（CLUE - TNEWS，15类
#中文新闻分类
raw = load_dataset("clue", "tnews")  # 公开数据集加载方式:contentReference[oaicite:1]{index=1}

#sentence(文本), label(标签)
texts = raw["train"]["sentence"][:8000]
labels = raw["train"]["label"][:8000]

# 划分训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

num_labels = len(set(labels))

#加载你本地中文 bert-base 模型
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    './models/google-bert/bert-base-chinese',
    num_labels=num_labels
)

#编码
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
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
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
eval_result = trainer.evaluate()
print("测试集评估：", eval_result)

#新样本测试
new_text = "苹果公司发布了新款手机，市场反响热烈。"
inputs = tokenizer(new_text, return_tensors="pt", truncation=True, padding=True, max_length=64)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

print("新样本：", new_text)
print("预测类别编号：", pred)
