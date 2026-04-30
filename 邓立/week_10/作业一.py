from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
import os

# 设置环境变量来禁用安全检查
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


def load_model():
    print("加载CLIP模型...")
    # 先创建模型配置
    config = ChineseCLIPModel.config_class.from_pretrained("/Users/tcl/Desktop/作业/nlp-study/凌云杰/week10/models/AI-ModelScope/chinese-clip-vit-base-patch16/")
    # 创建模型实例
    model = ChineseCLIPModel(config)
    # 加载状态字典
    state_dict = torch.load("/Users/tcl/Desktop/作业/nlp-study/凌云杰/week10/models/AI-ModelScope/chinese-clip-vit-base-patch16/pytorch_model.bin", map_location='cpu')
    # 将状态字典加载到模型中（忽略额外的键）
    model.load_state_dict(state_dict, strict=False)
    # 获取processor
    processor = ChineseCLIPProcessor.from_pretrained("/Users/tcl/Desktop/作业/nlp-study/凌云杰/week10/models/AI-ModelScope/chinese-clip-vit-base-patch16/")
    return model, processor


def encode_image(model, processor, image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.numpy()
        image_features = normalize(image_features)

    return image_features


def encode_text(model, processor, labels):
    inputs = processor(text=labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
        text_features = normalize(text_features)

    return text_features


def zero_shot_classification(image_path, labels):
    model, processor = load_model()

    image_features = encode_image(model, processor, image_path)
    text_features = encode_text(model, processor, labels)

    similarities = np.dot(image_features, text_features.T)
    sorted_indices = similarities.argsort()[0][::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            "label": labels[idx],
            "similarity": float(similarities[0][idx])
        })

    return results


if __name__ == "__main__":
    image_path = "/Users/tcl/Desktop/ai_study/第10周：多模态大模型/xiaogouzaicaopingshangxixi_13330961.jpg"

    labels = [
        "一只小狗",
        "一只小猫",
        "一只兔子",
        "一只小鸟",
        "一只松鼠",
        "一只熊猫",
        "一只老虎",
        "一只狮子",
        "一只猴子",
        "一只大象",
        "一片草地",
        "一片森林",
        "一只狗在草地上奔跑",
        "一只小狗在草地上玩耍",
        "一只宠物狗"
    ]

    results = zero_shot_classification(image_path, labels)

    print("CLIP零样本分类结果：")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label']}: {result['similarity']:.4f}")
    print("=" * 50)