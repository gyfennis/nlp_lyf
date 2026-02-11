#使用今日头条新闻文本数据集进行本次实践
#首先安装必要的库，然后加载和预处理数据
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 数据预处理示例
def preprocess_data(texts, labels, max_length=128):
    inputs = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs, torch.tensor(labels.tolist())

# 加载数据集（以今日头条数据集为例）
# 假设数据已整理为DataFrame，包含'text'和'label'两列
# df = pd.read_csv('toutiao_data.csv')
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
#模型加载与微调
# 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=15  # 今日头条数据集有15个类别
)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练循环
model.train()
for epoch in range(3):  # 训练3个epoch
    total_loss = 0
    for i in range(0, len(train_texts), 32):  # 批量处理
        batch_texts = train_texts[i:i+32]
        batch_labels = train_labels[i:i+32]

        inputs, labels = preprocess_data(batch_texts, batch_labels)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')
    #模型评估与预测
    def evaluate_model(model, texts, labels):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i+32]
            batch_labels = labels[i:i+32]

            inputs, labels_tensor = preprocess_data(batch_texts, batch_labels)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_tensor = labels_tensor.to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels_tensor).sum().item()
            total += len(labels_tensor)

    accuracy = correct / total
    print(f'测试准确率: {accuracy:.4f}')
    return accuracy

# 评估模型
# accuracy = evaluate_model(model, val_texts, val_labels)
#新样本测试验证
def predict_new_sample(model, text):
    model.eval()

    # 预处理输入文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return prediction.cpu().numpy(), probabilities.cpu().numpy()

# 类别映射（示例）
class_mapping = {
    0: "财经", 1: "科技", 2: "体育", 3: "娱乐",
    4: "时政", 5: "游戏", 6: "时尚", 7: "家居",
    8: "房产", 9: "教育", 10: "汽车", 11: "旅游",
    12: "职场", 13: "美食", 14: "文化"
}

# 测试新样本
new_text = "国际足球冠军赛昨晚落幕，梅西带领球队获得最终胜利"
predicted_class, probabilities = predict_new_sample(model, new_text)

print(f"输入文本: {new_text}")
print(f"预测类别: {class_mapping[predicted_class[0]]}")
print(f"各类别概率: {probabilities}")

