# 导入必要库
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载本地数据集
dataset_df = pd.read_csv('../week04/custom_dataset.csv')

# 查看数据结构
print("数据集形状:", dataset_df.shape)
print("列名:", dataset_df.columns)
print("前5行数据:")
print(dataset_df.head())

# 数据预处理
# 假设数据集中有 'text' 列和 'label' 列
texts = dataset_df['text'].tolist()
labels = dataset_df['label'].tolist()

# 编码标签
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    texts, encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

# 加载BERT分词器和模型
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_)
)

# 数据编码
def encode_data(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels)
    }

train_encodings = encode_data(x_train, y_train)
test_encodings = encode_data(x_test, y_test)

# 创建Dataset对象
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TextDataset(train_encodings)
test_dataset = TextDataset(test_encodings)

# 定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
print("开始训练...")
trainer.train()

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"测试集准确率: {eval_results['eval_accuracy']:.4f}")

# 测试新样本
def predict_text(text):
    # 编码输入文本
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )

    # 将输入张量移动到与模型相同的设备上
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(predictions, dim=-1).item()

    # 解码预测结果
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = predictions[0][predicted_class_idx].item()

    return predicted_label, confidence

# 测试样本
test_samples = [
    "这家餐厅的服务态度很好，菜品也很美味",
    "产品质量很差，完全不值得购买",
    "物流速度很快，包装也很仔细"
]

print("\n测试新样本:")
for sample in test_samples:
    pred_label, confidence = predict_text(sample)
    print(f"文本: {sample}")
    print(f"预测类别: {pred_label}")
    print(f"置信度: {confidence:.4f}")
    print("-" * 50)
