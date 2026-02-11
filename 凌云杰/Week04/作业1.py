import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


df = pd.read_csv("news.csv")
print("数据集类别分布：")
print(df["label"].value_counts())

# 标签编码：文本类别 → 数字
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# 划分训练集 / 测试集
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)


tokenizer = BertTokenizer.from_pretrained("models/google-bert/bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

# 构造 HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# 只保留模型需要的列
train_dataset = train_dataset.remove_columns(["text", "label", "__index_level_0__"]).rename_column("label_id", "labels")
test_dataset = test_dataset.remove_columns(["text", "label", "__index_level_0__"]).rename_column("label_id", "labels")

# -------------------------- 3. 评估指标 --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# -------------------------- 4. 模型与训练配置 --------------------------
model = BertForSequenceClassification.from_pretrained(
    "models/google-bert/bert-base-chinese",
    num_labels=len(label_encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./bert_small_news",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -------------------------- 5. 开始训练 --------------------------
print("\n开始训练...")
trainer.train()

# 最终评估
eval_result = trainer.evaluate()
print(f"\n测试集准确率: {eval_result['eval_accuracy']:.4f}")

# -------------------------- 6. 新样本预测函数 --------------------------
def predict_single_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    return pred_label

# -------------------------- 7. 测试新样本 --------------------------
if __name__ == "__main__":
    test_samples = [
        "国产芯片性能大幅提升打破国外垄断",
        "英超联赛曼联客场2-1逆转利物浦",
        "央行加大对小微企业信贷投放力度",
        "暑期档动画电影票房连续破亿"
    ]

    print("\n===== 新样本预测结果 =====")
    for s in test_samples:
        pred = predict_single_text(s)
        print(f"文本: {s}")
        print(f"预测类别: {pred}\n")