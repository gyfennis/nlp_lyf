import os
import base64
from io import BytesIO
from openai import OpenAI
import fitz  # PyMuPDF

def pdf_to_image(pdf_path, page_num=0, dpi=144):
    """
    将 PDF 的指定页面转换为图片

    Args:
        pdf_path: PDF 文件路径
        page_num: 页码（从 0 开始）
        dpi: 分辨率，默认 144

    Returns:
        PIL Image 对象
    """
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(f"页码超出范围，PDF 共有{len(doc)}页")

    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    img_data = pix.tobytes("png")

    from PIL import Image
    img = Image.open(BytesIO(img_data))

    doc.close()
    return img


def image_to_base64(image):
    """
    将 PIL Image 转换为 base64 编码
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


def analyze_pdf_with_qwen(pdf_path, api_key, prompt=None):
    """
    使用 Qwen3-VL-Plus 模型分析 PDF 第一页

    Args:
        pdf_path: PDF 文件路径
        api_key: 阿里云 API Key
        prompt: 提示词，默认为通用解析

    Returns:
        模型返回的分析结果
    """
    if prompt is None:
        prompt = "请详细分析这张文档的内容，包括：\n1. 文档标题和作者\n2. 主要内容概述\n3. 关键技术和方法\n4. 重要图表说明\n请用中文回答。"

    print("正在读取 PDF 第一页...")
    img = pdf_to_image(pdf_path, page_num=0)
    print(f"图片尺寸：{img.size}")

    base64_image = image_to_base64(img)

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    print("正在调用 Qwen3-VL-Plus 模型进行分析...")

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=2048
    )

    result = completion.choices[0].message.content
    return result


if __name__ == "__main__":
    pdf_path = r"E:\BaiduNetdiskDownload\nlp\Week04\论文\2017 Attention Is All You Need.pdf"

    api_key = os.getenv("aliyun_APIkey")

    if not api_key:
        print("错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("请在阿里云百炼平台获取 API Key：https://help.aliyun.com/zh/model-studio/get-api-key")
        print("然后设置环境变量：set DASHSCOPE_API_KEY=your_api_key_here")
        exit(1)

    try:
        result = analyze_pdf_with_qwen(pdf_path, api_key)
        print("\n" + "="*50)
        print("PDF 分析结果：")
        print("="*50)
        print(result)

    except Exception as e:
        print(f"发生错误：{e}")
