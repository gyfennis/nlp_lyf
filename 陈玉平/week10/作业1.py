from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

# 官方 openai clip 不支持中文
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
model = ChineseCLIPModel.from_pretrained("D:\code\LLMModels\chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("D:\code\LLMModels\chinese-clip-vit-base-patch16") # 预处理


# 2. 加载本地图片
image_path = "D:/code/gitcode/hub-QdXP/OIP-C.jpg"  # 替换为你的图片路径
image = Image.open(image_path)

# 3. 定义候选类别（Zero-Shot 的关键！）
# 用自然语言描述，可以非常灵活
candidate_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a photo of a person",
    "a photo of food"
]

# 4. 预处理
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 5. 前向传播
with torch.no_grad():
    outputs = model(**inputs)

    # 获取图像和文本的特征
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    # 计算相似度（余弦相似度）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T) * 100  # 缩放因子
    probs = similarity.softmax(dim=-1)

# 6. 输出结果
for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob.item():.4f} ({prob.item() * 100:.2f}%)")

# 获取最佳预测
best_idx = probs.argmax().item()
print(f"\n预测结果: {candidate_labels[best_idx]} (置信度: {probs[0][best_idx].item() * 100:.2f}%)")