import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertConfig
)
from datasets import Dataset
import warnings
import json

warnings.filterwarnings('ignore')


# 设置随机种子保证可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# 1. 加载和预处理数据 - 主要修改这里
# 从JSONL文件读取数据
texts = []
labels = []

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        # 将instruction作为输入文本
        texts.append(data["instruction"])
        # output作为标签
        labels.append(data["output"])

# 使用LabelEncoder将文本标签转换为数字
lbl = LabelEncoder()
labels_encoded = lbl.fit_transform(labels)

print(f"数据集大小: {len(texts)}")
print(f"类别数量: {len(set(labels))}")

# 分割数据
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")

# 2. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(r'D:\course\code\week6-bert-base-chinese')

# 使用配置初始化模型
config = BertConfig.from_pretrained(
    r'D:\course\code\week6-bert-base-chinese',
    num_labels=len(set(labels)),
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

model = BertForSequenceClassification.from_pretrained(
    r'D:\course\code\week6-bert-base-chinese',
    config=config
)


# 3. 创建Dataset（使用优化的方式）
def preprocess_data(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=128,  # 增加长度
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist(),
        'labels': labels.tolist() if isinstance(labels, np.ndarray) else labels
    })

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


train_dataset = preprocess_data(x_train, train_labels)
test_dataset = preprocess_data(x_test, test_labels)


# 4. 定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}


# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=8,  # 减小batch size
    per_device_eval_batch_size=8,
    learning_rate=2e-5,  # 添加学习率
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    gradient_accumulation_steps=2,  # 模拟更大的batch size
    fp16=True if torch.cuda.is_available() else False,  # 使用混合精度训练
    dataloader_num_workers=0,  # Windows上设置为0避免multiprocessing问题
)

# 6. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. 训练和评估
print("开始训练...")
train_result = trainer.train()
print(f"训练完成，损失: {train_result.metrics['train_loss']:.4f}")

print("开始评估...")
eval_result = trainer.evaluate()
print(f"评估结果 - 准确率: {eval_result['eval_accuracy']:.4f}, 损失: {eval_result['eval_loss']:.4f}")

# 8. 保存模型
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
print("模型已保存到 ./best_model")

# 9. 预测示例
print("\n预测示例:")
sample_texts = x_test[:3]

# 10. 新增：测试新样本
print("\n测试新样本:")
new_samples = [
    "请帮我备注不要送货上门",
    "这个商品坏了，我要退款",
    "可以开发票吗？"
]

# 将模型移到CPU
model.to('cpu')


# 定义预测函数
def predict_texts(texts):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encodings)
        # 修改这里：使用outputs[0]而不是outputs.logits
        predictions = torch.argmax(outputs[0], dim=-1)

    return predictions


# 预测示例
sample_preds = predict_texts(sample_texts)
for i, (text, pred) in enumerate(zip(sample_texts, sample_preds)):
    print(f"文本{i + 1}: {text[:50]}...")
    print(f"预测类别: {pred.item()}, 真实类别: {test_labels[i]}")
    print(f"预测标签: {lbl.inverse_transform([pred.item()])[0]}, 真实标签: {lbl.inverse_transform([test_labels[i]])[0]}")
    print("-" * 50)

# 预测新样本
new_preds = predict_texts(new_samples)
for i, (text, pred) in enumerate(zip(new_samples, new_preds)):
    predicted_label = lbl.inverse_transform([pred.item()])[0]
    print(f"新样本{i + 1}: {text}")
    print(f"预测类别: {predicted_label}")
    print("-" * 50)