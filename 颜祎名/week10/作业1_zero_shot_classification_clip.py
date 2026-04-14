import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- 1. 模型和处理器加载 ---
# 我们使用 OpenAI 的原始 CLIP 模型，它非常轻量且高效
# 'openai/clip-vit-base-patch32' 是一个常用且效果不错的版本
model_id = "openai/clip-vit-base-patch32"

print(f"正在从 Hugging Face 加载 CLIP 模型 '{model_id}'，请稍候...")
# 加载预训练的 CLIP 模型
model = CLIPModel.from_pretrained(model_id)
# 加载对应的处理器，它会负责将图片和文本转换成模型需要的格式
processor = CLIPProcessor.from_pretrained(model_id)
print("模型加载完成！")


# --- 2. 准备输入 ---
# 指定你的本地图片路径
image_path = "dog.webp"

# 检查图片是否存在
if not os.path.exists(image_path):
    print(f"错误：图片文件 '{image_path}' 不存在。请确保图片已放置在项目根目录下。")
    exit()

# 打开图片
try:
    image = Image.open(image_path)
except Exception as e:
    print(f"错误：无法打开或处理图片文件 '{image_path}'。错误信息: {e}")
    exit()

# 定义零样本分类的候选标签
candidate_labels = ["cat", "dog", "bird", "car", "house"]
print(f"\n候选标签: {candidate_labels}")


# --- 3. 模型推理 ---
# 这是 CLIP 的核心步骤
print("\n正在使用 CLIP 模型进行推理...")

# 使用处理器来准备输入。它会同时处理图片和所有文本标签。
# padding=True 和 return_tensors="pt" 是标准做法
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 模型会分别计算图片特征和文本特征，然后计算它们之间的相似度分数（logits）
# logits_per_image 的形状是 (图片数量, 文本标签数量)
with torch.no_grad(): # 在推理时关闭梯度计算，可以节省资源
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

# --- 4. 解析并输出结果 ---
# logits 本身是原始分数，我们可以使用 softmax 将其转换为概率
# 这能让我们更直观地看到模型对每个标签的“置信度”
probs = logits_per_image.softmax(dim=1)

# 找到概率最高的那个标签的索引
highest_prob_index = probs.argmax().item()
predicted_label = candidate_labels[highest_prob_index]

print("\n--- 作业结果 ---")
print("模型计算出的每个标签的概率分布：")
for i, label in enumerate(candidate_labels):
    print(f"- {label}: {probs[0, i].item():.4f} ({probs[0, i].item()*100:.2f}%)")

print("\n--- 最终结论 ---")
print(f"根据 CLIP 模型的计算，这张图片最可能被分类为: 【{predicted_label}】")
