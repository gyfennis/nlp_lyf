import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np


# 加载和预处理数据
dataset_df = pd.read_csv("2.csv")

# 打印前五行数据集
print(dataset_df.head(5))

# 创建标签编码器，将标签转换为数字标签
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df["label"].values)
text = list(dataset_df['text'].values)

x_train, x_test, train_labels, test_labels = train_test_split(
    text,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels,    # 确保训练集和测试集的标签分布一致
    random_state=326
)

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(r'E:\BaiduNetdiskDownload\nlp\models\google-bert\bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(r'E:\BaiduNetdiskDownload\nlp\models\google-bert\bert-base-chinese', num_labels=3)

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_encodings = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
 })

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建 Trainer 对象，简化模型训练代码
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_encodings,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()