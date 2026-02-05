import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


id2label = {
    0: '财经', 1: '房产', 2: '股票', 3: '教育', 4: '科技',
    5: '社会', 6: '时政', 7: '体育', 8: '游戏', 9: '娱乐'
}


model_path = "./cnews_results/checkpoint-750"

print(f"正在从本地加载模型和分词器: {model_path} ...")

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(id2label))
except OSError:
    print(f" 错误：找不到路径 {model_path}")
    print("如果是分词器文件缺失，请把 tokenizer = ... 那行改回 'bert-base-chinese'")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]


test_cases = [
    "詹姆斯砍下30分助湖人逆转勇士。",
    "上证指数今日收盘下跌1.5%，白酒板块领跌。",
    "装修时一定要注意甲醛检测，多通风。",
    "OpenAI发布的Sora模型可以生成高清视频。",
    "教育部发布通知，中小学将减轻作业负担。",
    "魔兽世界怀旧服即将开启新的团本。",
    "杨幂新剧收视率破亿，演技备受争议。"
]

print("="*30)
print("    最终模型预测结果   ")
print("="*30)

for text in test_cases:
    category = predict(text)
    print(f" 文本: {text}")
    print(f" 预测: 【{category}】")
    print("-" * 20)
