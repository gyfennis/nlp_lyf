import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,logging
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
logging.set_verbosity_error()

dataset_df = pd.read_csv("./Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv", sep=",", header=None)
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df[1].values[:500])
texts = list(dataset_df[0].values[:500])

x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)
#
#
#
#
tokenizer = BertTokenizer.from_pretrained("/Users/arvin/Desktop/AI/models/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("/Users/arvin/Desktop/AI/models/bert-base-chinese", num_labels=17)


train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=40)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=40)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})
#
#
#
#
#
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=10,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)
#
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

trainer.train()
trainer.evaluate()

device = next(model.parameters()).device
new_text = "我今天有点不舒服。"
new_input=tokenizer(new_text, return_tensors="pt", truncation=True, max_length=40, padding=True)
new_input = {k: v.to(device) for k, v in new_input.items()}
with torch.no_grad():
    new_outputs = model(**new_input)
new_pred = torch.argmax(new_outputs.logits, dim=-1).item()
print(lbl.inverse_transform([new_pred])[0])