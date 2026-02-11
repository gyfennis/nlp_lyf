#环境配置与项目设置
# requirements.txt
torch>=1.9.0
transformers>=4.20.0
flask>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
#意图识别数据集准备
import pandas as pd
from sklearn.model_selection import train_test_split

# 创建意图识别数据集（3个以上类别）
intent_data = [
    {"text": "打开客厅的灯", "label": "device_control"},
    {"text": "关闭空调", "label": "device_control"},
    {"text": "今天天气怎么样", "label": "weather_query"},
    {"text": "明天会下雨吗", "label": "weather_query"},
    {"text": "播放周杰伦的歌", "label": "music_control"},
    {"text": "暂停音乐", "label": "music_control"},
    {"text": "设置明天早上7点的闹钟", "label": "alarm_set"},
    {"text": "提醒我下午开会", "label": "reminder_set"},
    {"text": "讲个笑话", "label": "entertainment"},
    {"text": "新闻头条", "label": "news_query"}
]

# 转换为DataFrame
df = pd.DataFrame(intent_data)
print("数据集分布：")
print(df['label'].value_counts())
#基于BERT的意图识别模型
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 创建标签映射
        self.label2id = {label: idx for idx, label in enumerate(set(labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.label2id[label], dtype=torch.long)
        }

class IntentClassifier:
    def __init__(self, model_name='bert-base-chinese', num_labels=6):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 标签映射（在实际应用中应该从训练数据中获取）
        self.label_mapping = {
            0: "device_control",
            1: "weather_query",
            2: "music_control",
            3: "alarm_set",
            4: "reminder_set",
            5: "entertainment"
        }

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 验证集评估
            val_accuracy = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        return correct / total

    def predict(self, text):
        self.model.eval()

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        predicted_label = self.label_mapping[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()

        return {
            'intent': predicted_label,
            'confidence': confidence,
            'probabilities': {self.label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
        #模型训练与验证
        def train_and_evaluate():
    # 准备数据
    texts = [item['text'] for item in intent_data]
    labels = [item['label'] for item in intent_data]

    # 分割训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 初始化分类器
    classifier = IntentClassifier(num_labels=6)

    # 创建数据集
    train_dataset = IntentDataset(train_texts, train_labels, classifier.tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, classifier.tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # 训练模型
    classifier.train(train_loader, val_loader, epochs=3)

    return classifier

# 训练模型
print("开始训练意图识别模型...")
classifier = train_and_evaluate()
print("模型训练完成！")

# 测试新样本
test_samples = [
    "打开卧室的灯",
    "今天温度多少度",
    "播放一些轻音乐",
    "设置明天早上的闹钟"
]

print("\n=== 新样本测试 ===")
for sample in test_samples:
    result = classifier.predict(sample)
    print(f"输入: {sample}")
    print(f"预测意图: {result['intent']} (置信度: {result['confidence']:.4f})")
    print("---")
#本地部署与API服务
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# 全局模型实例
classifier_instance = None

def initialize_model():
    """初始化模型（在实际应用中应该加载已训练好的模型）"""
    global classifier_instance
    classifier_instance = IntentClassifier(num_labels=6)
    # 这里应该加载训练好的权重
    print("模型初始化完成")

@app.route('/predict', methods=['POST'])
def predict_intent():
    """意图预测接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = classifier_instance.predict(text)

        return jsonify({
            'status': 'success',
            'input_text': text,
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'model_loaded': classifier_instance is not None})

def start_flask_app():
    """启动Flask应用"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# 启动服务
if __name__ == '__main__':
    print("正在初始化意图识别模型...")
    initialize_model()

    print("启动Flask API服务...")
    print("服务地址: http://localhost:5000")
    print("可用接口:")
    print("  GET  /health    - 健康检查")
    print("  POST /predict   - 意图预测")

    start_flask_app()
#API调用测试
import requests
import json

def test_api():
    """测试API接口"""
    base_url = "http://localhost:5000"

    # 健康检查
    try:
        response = requests.get(f"{base_url}/health")
        print("健康检查:", response.json())
    except Exception as e:
        print(f"服务连接失败: {e}")
        return

    # 测试样本
    test_cases = [
        "关闭空调",
        "明天天气怎么样",
        "播放我最喜欢的歌曲",
        "提醒我晚上八点开会"
    ]

    print("\n=== API测试结果 ===")
    for text in test_cases:
        payload = {"text": text}

        try:
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                print(f"输入: {text}")
                print(f"意图: {prediction['intent']} (置信度: {prediction['confidence']:.4f})")
            else:
                print(f"请求失败: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"请求异常: {e}")

        print("---")

# 在另一个进程中运行测试
if __name__ == '__main__':
    # 等待服务启动
    import time
    time.sleep(3)
    test_api()
#部署优化建议
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
