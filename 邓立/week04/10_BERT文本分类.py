import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据（读取本地 Markdown 数据集）
# dataset_markdown.md 表格包含 text 和 label 两列
md_path = "dataset_markdown.md"

with open(md_path, "r", encoding="utf-8") as f:
    md_lines = [line.strip() for line in f.readlines()]

# 解析 Markdown 表格
rows = []
for line in md_lines:
    if not line.startswith("|"):
        continue
    # 跳过分隔行
    if set(line.replace("|", "").strip()) <= set("- "):
        continue
    parts = [p.strip() for p in line.strip("|").split("|")]
    if len(parts) != 2:
        continue
    if parts[0] == "text" and parts[1] == "label":
        continue
    rows.append(parts)

if len(rows) == 0:
    raise ValueError("未在 dataset_markdown.md 中发现可用数据行")

dataset_df = pd.DataFrame(rows, columns=["text", "label"])

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df["label"].values)
texts = list(dataset_df["text"].values)

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.25,    # 测试集比例为25%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)


# 从预训练模型加载分词器和模型
# num_labels 由当前数据集类别数决定
num_labels = len(lbl.classes_)
model_path = "../bert_base_models"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
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


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=8,       # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=8,        # 评估时每个设备的批次大小
    warmup_steps=50,                     # 学习率预热的步数
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=20,                    # 每隔20步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

# 新样本测试（可直接修改文本）
new_text = "公司宣布新产品发布，股价应声上涨。"
new_enc = tokenizer([new_text], truncation=True, padding=True, max_length=64, return_tensors="pt")
new_enc = {k: v.to(model.device) for k, v in new_enc.items()}

with torch.no_grad():
    outputs = model(**new_enc)
    pred_label = torch.argmax(outputs.logits, dim=-1).item()

pred_label_name = lbl.inverse_transform([pred_label])[0]
print(f"[predict] text={new_text}")
print(f"[predict] label_id={pred_label}, label_name={pred_label_name}")
