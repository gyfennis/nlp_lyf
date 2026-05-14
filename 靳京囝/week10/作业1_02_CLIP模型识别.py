from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

# 本地模型路径
model_path = r"E:\Develop\1_Core_Environments\modelscope\models\AI-ModelScope\chinese-clip-vit-base-patch16"

# 加载模型
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

# 图片 + 分类标签
image = Image.open("./作业1_01_小狗图片.jpg").convert("RGB")
texts = ["小狗", "小猫", "汽车", "树木"]

# 推理
inputs = processor(images=image, text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1)

# 输出
idx = probs.argmax().item()
print(f"预测：{texts[idx]}，置信度：{probs[idx]:.2f}")