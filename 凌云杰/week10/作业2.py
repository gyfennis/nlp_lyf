#!/usr/bin/env python3
"""
使用云端Qwen-VL模型解析本地PDF第一页

依赖：
- PyMuPDF (fitz) - 用于PDF转图片
- Pillow (PIL) - 用于图像处理
- openai - 用于API调用

安装依赖：
pip install pymupdf pillow openai
"""

import os
import base64
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

def pdf_to_image(pdf_path, page_number=0):
    """将PDF的指定页面转换为图片"""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def image_to_base64(image):
    """将图片转换为base64编码"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def parse_pdf_with_qwen_vl(pdf_path, api_key):
    """使用Qwen-VL模型解析PDF第一页"""
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 将PDF第一页转换为图片
    image = pdf_to_image(pdf_path)
    
    # 将图片转换为base64
    image_base64 = image_to_base64(image)
    
    # 创建聊天完成请求
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    },
                },
                {"type": "text", "text": "请解析这张PDF页面的内容，包括所有文本和结构信息。"},
            ],
        },
    ]
    
    completion = client.chat.completions.create(model="qwen3-vl-plus",  messages=messages,stream=True)
    
    print("=" * 20 + "Qwen-VL解析结果" + "=" * 20 + "\n")
    
    full_response = ""
    for chunk in completion:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end='', flush=True)
                full_response += delta.content
    
    print("\n" + "=" * 50)
    return full_response

if __name__ == "__main__":
    # PDF文件路径
    pdf_path = "/Users/tcl/Desktop/作业/nlp-study/凌云杰/week10/应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"
    
    # API Key
    api_key = "sk-2123e2a31d89476。。。。。。"
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误：PDF文件不存在：{pdf_path}")
    else:
        # 解析PDF
        parse_pdf_with_qwen_vl(pdf_path, api_key)
