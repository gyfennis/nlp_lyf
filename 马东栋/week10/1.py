#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese CLIP 零样本图像分类
"""

import argparse
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
from matplotlib import pyplot as plt


def load_model(model_path):
    """
    加载 Chinese CLIP 模型和处理器
    
    Args:
        model_path: 模型路径
        
    Returns:
        model: 加载的模型
        processor: 加载的处理器
        device: 使用的设备
    """
    print(f"正在加载模型: {model_path}")
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"模型加载完成，使用设备: {device}")
    
    return model, processor, device


def load_image(image_path):
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        image: 加载的图像
    """
    try:
        image = Image.open(image_path)
        print(f"图像加载成功: {image_path}")
        return image
    except Exception as e:
        print(f"图像加载失败: {e}")
        raise


def zero_shot_classification(model, processor, image, candidate_labels, device):
    """
    零样本图像分类
    
    Args:
        model: Chinese CLIP 模型
        processor: Chinese CLIP 处理器
        image: 输入图像
        candidate_labels: 候选类别标签
        device: 使用的设备
        
    Returns:
        predicted_label: 预测的类别
        probabilities: 所有类别的概率
        candidate_labels: 候选类别标签
    """
    # 图像编码
    inputs_image = processor(images=image, return_tensors="pt")
    # 将输入移到正确的设备
    for key in inputs_image:
        inputs_image[key] = inputs_image[key].to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 文本编码
    inputs_text = processor(text=candidate_labels, return_tensors="pt", padding=True)
    # 将输入移到正确的设备
    for key in inputs_text:
        inputs_text[key] = inputs_text[key].to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity = (image_features @ text_features.T).squeeze()
    print(f"相似度计算完成，形状: {similarity.shape}")
    
    # 获取预测结果
    probabilities = similarity.softmax(dim=0)
    predicted_label_idx = probabilities.argmax().item()
    predicted_label = candidate_labels[predicted_label_idx]
    
    return predicted_label, probabilities, candidate_labels


def display_results(predicted_label, probabilities, candidate_labels):
    """
    显示分类结果
    
    Args:
        predicted_label: 预测的类别
        probabilities: 所有类别的概率
        candidate_labels: 候选类别标签
    """
    # 找到预测类别的索引
    predicted_label_idx = candidate_labels.index(predicted_label)
    
    print("\n" + "="*60)
    print("=== Zero-Shot Classification 结果 ===")
    print("="*60)
    print(f"预测类别：{predicted_label}")
    print(f"置信度：{probabilities[predicted_label_idx].item():.4f}")
    
    print("\n=== 所有类别的概率分布 ===")
    sorted_results = sorted(zip(candidate_labels, probabilities),
                           key=lambda x: x[1].item(),
                           reverse=True)
    
    for i, (label, prob) in enumerate(sorted_results):
        marker = "✓" if label == predicted_label else " "
        print(f"{marker} {i+1:2d}. {label:8s}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    print("="*60)


def visualize_results(image, predicted_label, probabilities, candidate_labels):
    """
    可视化分类结果
    
    Args:
        image: 输入图像
        predicted_label: 预测的类别
        probabilities: 所有类别的概率
        candidate_labels: 候选类别标签
    """
    # 获取 Top-3 结果
    top3_idx = probabilities.argsort(descending=True)[:3]
    
    plt.figure(figsize=(15, 6))
    
    # 左侧显示原图
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title(f"输入图片\n预测：{predicted_label}", fontsize=12, fontproperties='SimHei')
    plt.axis('off')
    
    # 右侧显示 Top-3 类别
    colors = ['#4CAF50', '#FF9800', '#F44336']  # 绿色、橙色、红色
    for i, idx in enumerate(top3_idx):
        plt.subplot(1, 4, i+2)
        bar_height = probabilities[idx].item()
        plt.barh([0], [bar_height], height=0.5, color=colors[i])
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.yticks([0], [candidate_labels[idx]], fontproperties='SimHei')
        plt.xlabel("概率")
        plt.title(f"Top-{i+1}: {probabilities[idx].item():.4f}", fontproperties='SimHei')
        # 隐藏不必要的边框
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Chinese CLIP 零样本图像分类")
    parser.add_argument('--image', type=str, default=r"E:\BaiduNetdiskDownload\nlp\Week10\Week10\dog.png",
                        help="输入图像路径")
    parser.add_argument('--model', type=str, default=r"E:\BaiduNetdiskDownload\nlp\models\chinese-clip-vit-base-patch16",
                        help="模型路径")
    args = parser.parse_args()
    
    # 定义候选类别标签
    candidate_labels = [
        "小狗", "小猫", "小鸟", "小兔", "小熊",
        "汽车", "飞机", "火车", "轮船", "自行车"
    ]
    
    try:
        # 加载模型
        model, processor, device = load_model(args.model)
        
        # 加载图像
        image = load_image(args.image)
        
        # 显示原始图像
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title("原始图像", fontproperties='SimHei')
        plt.axis('off')
        plt.show()
        
        # 零样本分类
        predicted_label, probabilities, candidate_labels = zero_shot_classification(
            model, processor, image, candidate_labels, device
        )
        
        # 显示结果
        display_results(predicted_label, probabilities, candidate_labels)
        
        # 可视化结果
        visualize_results(image, predicted_label, probabilities, candidate_labels)
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
