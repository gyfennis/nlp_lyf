from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载分词器（保持不变）
tokenizer = BertTokenizer.from_pretrained('E:\\BaiduNetdiskDownload\\nlp\\models\\google-bert\\bert-base-chinese')

# 加载训练好的模型
model = BertForSequenceClassification.from_pretrained(
    './results/checkpoint-96'
)

# 预测函数
def predict(text):
    # 编码输入
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64
    )

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(predictions, dim=-1).item()

    # 需要保存 LabelEncoder 才能还原标签
    return predicted_class_idx

# 测试
test_samples = [
    "什么是机器学习？",           # 应该预测：机器学习
    "你对线性回归了解多少？",     # 应该预测：机器学习
    "什么是神经网络？",           # 应该预测：深度学习
    "LLM 是什么？",              # 应该预测：LLM
    "深度学习和机器学习的区别",   # 应该预测：深度学习/机器学习
    "Transformer 架构的原理",    # 应该预测：LLM/深度学习
    "什么是大语言模型？",         # 应该预测：LLM
    "请解释一下监督学习"         # 应该预测：机器学习
]
for sample in test_samples:
    predicted_class = predict(sample)
    print(f"输入：{sample}，预测的标签索引：{predicted_class}")