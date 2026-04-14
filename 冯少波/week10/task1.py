#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP Zero-Shot 图像分类演示
使用本地小狗图片进行分类
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.preprocessing import normalize

# 配置
IMAGE_PATH = "/Users/test/Library/Application Support/Qoder/SharedClientCache/cache/images/f563f795/dog-5cdae5e5.jpg"
MODEL_PATH = "./model/chinese-clip-vit-base-patch16"

# Zero-Shot 分类标签（中英文）
CLASS_LABELS = [
    "一只狗",           # 正确标签
    "一只猫",
    "一只鸟",
    "一辆车",
    "一个人",
    "一座房子",
    "一棵树",
    "一朵花",
    "一只柯基犬",       # 更具体的标签
    "一只宠物",
]


def log_step(step_num, title):
    """打印步骤日志"""
    print(f"\n{'='*70}")
    print(f"【步骤 {step_num}】{title}")
    print('='*70)


def log_info(msg):
    """打印信息"""
    print(f"  [INFO] {msg}")


def log_success(msg):
    """打印成功"""
    print(f"  [OK] ✓ {msg}")


def load_model():
    """加载 CLIP 模型"""
    log_step(1, "加载 CLIP 模型")
    
    # 如果本地模型不存在，尝试使用 modelscope 缓存路径
    if not os.path.exists(MODEL_PATH):
        model_path = os.path.expanduser(
            "~/.cache/modelscope/hub/models/AI-ModelScope/chinese-clip-vit-base-patch16"
        )
    else:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在于 {model_path}")
        print("请下载模型: modelscope download --model AI-ModelScope/chinese-clip-vit-base-patch16")
        sys.exit(1)
    
    log_info(f"模型路径: {model_path}")
    
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    
    log_success("模型加载完成")
    return model, processor


def load_image(image_path):
    """加载并预处理图片"""
    log_step(2, "加载图片")
    
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 {image_path}")
        sys.exit(1)
    
    log_info(f"图片路径: {image_path}")
    
    # 加载图片
    image = Image.open(image_path)
    log_info(f"原始尺寸: {image.size}")
    log_info(f"图片模式: {image.mode}")
    
    # 转换为 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        log_info("已转换为 RGB 模式")
    
    # 显示图片信息
    log_success(f"图片加载成功")
    
    return image


def encode_image(image, processor, model):
    """编码图片为特征向量"""
    log_step(3, "图片编码")
    
    # 使用 processor 处理图片
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.numpy()
    
    # 归一化
    image_features = normalize(image_features)[0]
    
    log_success(f"图片特征维度: {image_features.shape}")
    return image_features


def encode_texts(labels, processor, model):
    """编码文本标签为特征向量"""
    log_step(4, "文本标签编码")
    
    log_info(f"分类标签: {labels}")
    
    # 批量编码所有标签
    inputs = processor(
        text=labels,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
    
    # 归一化
    text_features = normalize(text_features)
    
    log_success(f"文本特征维度: {text_features.shape}")
    return text_features


def zero_shot_classify(image_features, text_features, labels):
    """Zero-Shot 分类"""
    log_step(5, "Zero-Shot 分类")
    
    # 计算相似度（余弦相似度，因为已归一化，点积即余弦相似度）
    similarities = np.dot(text_features, image_features)
    
    # Softmax 转换为概率
    exp_similarities = np.exp(similarities)
    probabilities = exp_similarities / np.sum(exp_similarities)
    
    # 排序结果
    results = []
    for i, label in enumerate(labels):
        results.append({
            'label': label,
            'similarity': similarities[i],
            'probability': probabilities[i]
        })
    
    # 按相似度排序
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results


def display_results(results):
    """显示分类结果"""
    log_step(6, "分类结果")
    
    print("\n  排名 | 标签          | 相似度 | 概率")
    print("  " + "-" * 50)
    
    for i, result in enumerate(results, 1):
        label = result['label']
        similarity = result['similarity']
        probability = result['probability']
        
        # 标记最高分和正确答案
        marker = "⭐" if i == 1 else "  "
        if "狗" in label or "柯基" in label or "宠物" in label:
            marker += "🐕"
        else:
            marker += "  "
        
        print(f"  {i:>2}   | {label:<12} | {similarity:>6.3f} | {probability:>5.1%} {marker}")
    
    # 显示预测结果
    best_match = results[0]
    print(f"\n  🎯 预测结果: {best_match['label']}")
    print(f"     置信度: {best_match['probability']:.1%}")
    
    # 检查是否正确
    correct_labels = ["一只狗", "一只柯基犬", "一只宠物"]
    if best_match['label'] in correct_labels:
        print(f"     ✅ 预测正确！")
    else:
        print(f"     ⚠️ 预测可能不准确")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("CLIP Zero-Shot 图像分类演示")
    print("="*70)
    
    # 1. 加载模型
    model, processor = load_model()
    
    # 2. 加载图片
    image = load_image(IMAGE_PATH)
    
    # 3. 编码图片
    image_features = encode_image(image, processor, model)
    
    # 4. 编码文本标签
    text_features = encode_texts(CLASS_LABELS, processor, model)
    
    # 5. Zero-Shot 分类
    results = zero_shot_classify(image_features, text_features, CLASS_LABELS)
    
    # 6. 显示结果
    display_results(results)
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)


if __name__ == "__main__":
    main()
