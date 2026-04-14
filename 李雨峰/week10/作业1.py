import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import numpy as np

model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16")

image = Image.open("./dog.jpg")

candidate_labels = ["狗", "猫", "鸟", "鱼", "兔子", "马", "牛", "羊", "老虎", "狮子", "熊猫", "猴子"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).numpy()[0]

top5_idx = np.argsort(probs)[-5:][::-1]
print("预测结果:")
for idx in top5_idx:
    print(f"  {candidate_labels[idx]}: {probs[idx]:.4f}")