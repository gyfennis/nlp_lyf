import numpy as np
from PIL import Image
import requests
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import matplotlib.pyplot as plt
model = ChineseCLIPModel.from_pretrained('models/chinese-clip-vit-base-patch16/')
processor = ChineseCLIPProcessor.from_pretrained('models/chinese-clip-vit-basepatch16/')
image_path = "images.jpg"
img = [Image.open(image_path)]
input = processor(images=img, return_tensors='pt')
img_image_feat = []
with torch.no_grad():
 image_feature = model.get_image_features(**input)
 image_feature = image_feature.data.numpy()
 img_image_feat.append(image_feature)
img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)
img_texts_feat = []
texts = ['这是⼀只：⼩狗', '这是⼀只：⼩猫', '这是⼀只：⼩⻦', '这是⼀只：⻥', '这是⼀只：树']
inputs = processor(text=texts, return_tensors='pt', padding=True)
with torch.no_grad():
 text_features = model.get_text_features(**inputs)
 text_features = text_features.data.numpy()
 img_texts_feat.append(text_features)
img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)
print(img_texts_feat.shape)
sim_result = np.dot(img_image_feat[0], img_texts_feat.T)
sim_idx = sim_result.argsort()[::-1][0]


response = client.chat.completions.create(
 model='qwen-vl-max',
  messages=[
 {
 "role": "user",
 "content": [
 {"type": "text", "text": "将图⽚内容提取为 Markdown 格式，输出关键信息。"},
 {
 "type": "image_url",
 "image_url": {
 "url": f"data:image/jpeg;base64,{base64_image}"
 },
 },
 ],
 }
 ],
)
