import os
import base64
import fitz  # PyMuPDF
from openai import OpenAI


def pdf_page_to_base64(pdf_path: str, page_num: int = 0, dpi: int = 150):
    """将 PDF 指定页转换为 base64 编码的图片"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 提高分辨率以获得更清晰的文字
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # 转换为 PNG 并 base64 编码
    img_bytes = pix.tobytes("png")
    base64_str = base64.b64encode(img_bytes).decode('utf-8')

    doc.close()
    return f"data:image/png;base64,{base64_str}"


def extract_pdf_text(pdf_path: str, api_key: str = None):
    """
    使用 Qwen-VL 提取本地 PDF 的所有文字
    """
    # 初始化客户端
    client = OpenAI(
        api_key=api_key ,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 打开 PDF 获取页数
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    print(f"📄 PDF 共 {total_pages} 页，开始提取...")

    all_text = []

    # 逐页处理
    for page_num in range(total_pages):
        print(f"\n🔍 正在处理第 {page_num + 1}/{total_pages} 页...")

        # PDF 转 base64 图片
        image_base64 = pdf_page_to_base64(pdf_path, page_num)

        # 调用 Qwen-VL 识别文字
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 可换成 qwen-vl-max 或 qwen2.5-vl
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64  # 本地图片用 base64
                            },
                        },
                        {
                            "type": "text",
                            "text": "请提取这张图片中的所有文字内容，保持原有格式，不要遗漏。"
                        },
                    ],
                },
            ],
        )

        page_text = completion.choices[0].message.content
        all_text.append(f"=== 第 {page_num + 1} 页 ===\n{page_text}\n")
        print(f"✓ 第 {page_num + 1} 页完成")

    return "\n".join(all_text)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 安装依赖: pip install PyMuPDF openai

    # 配置 API Key
    API_KEY = "sk-c6625bb19dc448a7ab54d45902d6b5e3"  # 或直接用 "sk-xxx"

    # PDF 文件路径（改成你的本地 PDF）
    PDF_PATH = "C:/Users/chenzhuqi/Desktop/测试.pdf"

    # 提取文字
    result = extract_pdf_text(PDF_PATH, api_key=API_KEY)

    # 输出结果
    print("\n" + "=" * 50)
    print(result)

    # 保存到文件
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("\n✅ 已保存到 output.txt")