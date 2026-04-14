#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析工具 - 使用 LangChain Deep Agents SDK + Qwen-VL
支持多模态PDF内容解析，包括文本、表格、图片等
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import hashlib

# Deep Agents
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

# 配置
API_KEY = "sk-b2850aa9aff64528962998e0933d3912"
PDF_PATH = "/Users/test/Downloads/冯少波的资料/大模型培训/1、预习/第10周：多模态大模型/PDFParser/Week02-大模型使用与深度学习基础.pdf"
TARGET_PAGES = [4]  # 只解析第4页
OUTPUT_DIR = "./output/Week02_deepagent"

# LangSmith 配置
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = "pdf-parser-demo"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
DPI = 200


def pdf_page_to_image(pdf_path: str, page_num: int, output_dir: str, dpi: int = 200) -> str:
    """将PDF指定页转换为图片"""
    doc = fitz.open(pdf_path)
    page_idx = page_num - 1
    
    if page_idx >= len(doc):
        doc.close()
        return None
    
    page = doc[page_idx]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"page_{page_num:03d}.png")
    pix.save(image_path)
    
    doc.close()
    return image_path


def extract_images_from_page(pdf_path: str, page_num: int, output_dir: str) -> List[Dict]:
    """从PDF页面中提取内嵌图片"""
    doc = fitz.open(pdf_path)
    page_idx = page_num - 1
    
    if page_idx >= len(doc):
        doc.close()
        return []
    
    page = doc[page_idx]
    image_list = page.get_images()
    
    extracted_images = []
    img_output_dir = os.path.join(output_dir, "extracted_images")
    os.makedirs(img_output_dir, exist_ok=True)
    
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        image_filename = f"page{page_num}_img{img_index}_{image_hash}.{image_ext}"
        image_path = os.path.join(img_output_dir, image_filename)
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        extracted_images.append({
            "path": image_path,
            "filename": image_filename,
            "relative_path": f"extracted_images/{image_filename}",
            "index": img_index,
        })
    
    doc.close()
    return extracted_images


def image_to_base64(image_path: str) -> str:
    """将图片转为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============ Deep Agent Tools ============

def analyze_page_structure(image_path: str) -> str:
    """分析页面结构"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    image_base64 = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请分析这个PDF页面的结构和内容：1.判断页面类型（text/table/chart/image/mixed）2.识别页面元素（文字段落、图片、表格、图表等）3.描述每个图片/图表的位置和内容概要。请以JSON格式返回。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=2048,
        temperature=0.1
    )
    return response.choices[0].message.content


def extract_text_content(image_path: str) -> str:
    """提取页面文字内容"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    image_base64 = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请提取这个PDF页面中的所有文字内容：1.保留原有的段落结构和格式2.保留标题层级3.如果是表格，用Markdown表格格式输出4.对于图片和图表，在相应位置标注 [图片: 简要描述]。请直接输出提取的文字内容。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1
    )
    return response.choices[0].message.content


def describe_image(image_path: str) -> str:
    """描述图片内容"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    image_base64 = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容，包括：图片类型、主要内容、文字信息、数据含义等。请用中文回答。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=2048,
        temperature=0.1
    )
    return response.choices[0].message.content


def save_markdown(filename: str, content: str) -> str:
    """保存Markdown文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Saved to {filepath}"


def save_json(filename: str, data: dict) -> str:
    """保存JSON文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"Saved to {filepath}"


# ============ Main Function ============

def main():
    """主函数 - 使用Deep Agent解析PDF"""
    print(f"📄 PDF Deep Agent 解析器")
    print(f"🎯 目标: {Path(PDF_PATH).name} 第 {TARGET_PAGES} 页")
    print(f"📊 LangSmith 项目: {os.getenv('LANGSMITH_PROJECT')}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    
    # 获取PDF信息
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    doc.close()
    print(f"📊 PDF共 {total_pages} 页")
    
    # 创建Deep Agent
    print("\n🤖 初始化 Deep Agent...")
    agent = create_deep_agent(
        tools=[
            analyze_page_structure,
            extract_text_content,
            describe_image,
            save_markdown,
            save_json,
        ],
        system_prompt="""你是一个专业的PDF文档解析助手。你的任务是深度解析PDF文档的每一页内容。

请遵循以下工作流程：
1. 首先分析页面结构，识别内容类型和元素
2. 提取页面中的所有文字内容，保持原有格式
3. 对于页面中的每个图片，生成详细的描述
4. 将结果保存为Markdown和JSON格式

输出要求：
- Markdown文件必须包含图片链接，格式为 ![描述](路径)
- 文字内容和图片描述要完整准确
- 保存文件时使用提供的工具函数""",
        model=ChatOpenAI(
            model="qwen-vl-max",
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1,
            max_tokens=4096
        )
    )
    
    # 处理每一页
    results = []
    for page_num in TARGET_PAGES:
        if page_num > total_pages:
            print(f"⚠️ 页码 {page_num} 超出范围，跳过")
            continue
        
        print(f"\n📄 处理第 {page_num} 页...")
        
        # 转换页面为图片
        page_image = pdf_page_to_image(PDF_PATH, page_num, images_dir, DPI)
        
        # 提取内嵌图片
        extracted_images = extract_images_from_page(PDF_PATH, page_num, OUTPUT_DIR)
        print(f"   发现 {len(extracted_images)} 张内嵌图片")
        
        # 使用Deep Agent解析页面
        task = f"""请解析PDF第 {page_num} 页：

1. 分析页面结构：
   - 页面图片路径: {page_image}
   - 调用 analyze_page_structure 分析页面结构

2. 提取文字内容：
   - 调用 extract_text_content 提取文字

3. 描述内嵌图片：
   {chr(10).join([f"   - 图片 {img['index']}: {img['path']}" for img in extracted_images]) if extracted_images else "   - 无内嵌图片"}
   {chr(10).join([f"   - 调用 describe_image 描述图片 {img['path']}" for img in extracted_images]) if extracted_images else ""}

4. 生成Markdown文件：
   - 文件名: page_{page_num:03d}.md
   - 包含页面预览图链接: images/page_{page_num:03d}.png
   - 包含内嵌图片链接: {', '.join([img['relative_path'] for img in extracted_images]) if extracted_images else '无'}
   - 包含文字内容和图片描述
   - 使用 save_markdown 保存

请逐步执行以上任务，并返回解析结果的摘要。"""
        
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})        
        print(f"   ✅ 第 {page_num} 页解析完成")
        results.append({
            "page_number": page_num,
            "result": result
        })
    
    # 保存汇总索引
    print("\n💾 保存图文检索索引...")
    index_data = {
        "document": Path(PDF_PATH).name,
        "total_pages": total_pages,
        "parsed_pages": TARGET_PAGES,
        "output_dir": OUTPUT_DIR
    }
    save_json("index.json", index_data)
    
    print(f"\n✅ 全部完成！结果保存在: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
