import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置（固定，无需修改）=====================
# 类别映射（4分类）
label2id = {"股票": 0, "基金": 1, "期货": 2, "债券": 3}
id2label = {v: k for k, v in label2id.items()}
num_labels = 4  # 类别数
model_name = "bert-base-chinese"  # 加载中文BERT-base预训练模型
max_length = 128  # 文本最大长度（BERT默认最大512，短文本选128足够）
batch_size = 16  # 批次大小（CPU选8/16，GPU选32/64）
epochs = 3  # 训练轮数（小数据集3轮足够，避免过拟合）
save_path = "./bert_finance_classifier"  # 微调后模型保存路径

# ===================== 2. 数据加载与预处理 =====================
# 读取数据集
df = pd.read_csv("thucnews_finance_4cls.csv", encoding="utf-8")
# 划分训练集（80%）和测试集（20%）
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
# 转换为Hugging Face Dataset格式（适配Transformers库）
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# 文本分词函数（将文本转为模型可识别的input_ids/attention_mask）
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,  # 截断过长文本
        padding="max_length",  # 填充到max_length
        max_length=max_length
    )
# 对数据集进行分词处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 整理数据集格式（只保留模型需要的字段）
tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
# 数据填充器（动态padding，比固定padding更高效，可选）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===================== 3. 加载BERT-base预训练模型 =====================
# 加载BERT用于序列分类（num_labels指定分类数，自动加载预训练权重）
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
# 自动适配设备（CPU/GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===================== 4. 定义训练参数 =====================
training_args = TrainingArguments(
    output_dir="./bert_train_logs",  # 训练日志保存路径
    learning_rate=2e-5,  # BERT微调专用学习率（必须小，推荐2e-5/3e-5/5e-5）
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,  # 权重衰减，防止过拟合
    logging_dir="./logs",
    logging_steps=10,  # 每10步打印一次日志
    evaluation_strategy="epoch",  # 每轮训练后在测试集评估
    save_strategy="epoch",  # 每轮训练后保存模型
    load_best_model_at_end=True,  # 加载训练过程中最好的模型
    fp16=False,  # CPU设为False，GPU设为True加速
)

# ===================== 5. 定义评估函数（计算准确率）=====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ===================== 6. 模型训练与微调 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# 开始训练
trainer.train()
# 在测试集评估最终效果
eval_result = trainer.evaluate()
print(f"测试集准确率：{eval_result['eval_accuracy']:.4f}")

# ===================== 7. 保存微调后的模型 =====================
trainer.save_model(save_path)
print(f"微调后模型已保存至：{save_path}")

# ===================== 8. 新样本测试（核心：验证分类效果）=====================
def predict_text(text):
    """新样本预测函数：输入文本，输出分类结果"""
    # 分词处理
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    # 模型推理（关闭梯度计算，加速）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取预测结果
    logits = outputs.logits
    pred_label_id = torch.argmax(logits, axis=-1).item()
    pred_label = id2label[pred_label_id]
    return pred_label

# 测试3个新样本（覆盖不同类别，验证效果）
new_samples = [
    "上证指数今日上涨2.5%，科技股领涨，创业板指创年内新高",
    "易方达新基金发行规模突破百亿，债基和股基成投资者首选",
    "沪铜期货主力合约价格下跌，受全球大宗商品需求疲软影响",
    "十年期国债收益率持续走低，央行逆回购操作稳定市场流动性"
]
# 遍历预测并打印结果
print("\n===== 新样本测试结果 =====")
for idx, sample in enumerate(new_samples):
    pred = predict_text(sample)
    print(f"样本{idx+1}：{sample}")
    print(f"预测类别：{pred}\n")
