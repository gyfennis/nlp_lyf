import os
import base64
import io
import fitz
from PIL import Image
from dashscope import MultiModalConversation

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-399b434c3f5b4329a4600ec76ce4f7cc")


def pdf_first_page_to_base64(pdf_path, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    doc.close()
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_pdf(pdf_path):
    img_base64 = pdf_first_page_to_base64(pdf_path)

    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/png;base64,{img_base64}"},
            {"text": "请分析这张图片的内容：1.提取所有文字 2.描述图片主题和布局 3.总结核心信息"}
        ]
    }]

    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
        api_key=dashscope.api_key
    )

    result = response.output.choices[0].message.content[0]["text"]
    return result


if __name__ == "__main__":
    pdf_path = "./Week10-多模态大模型.pdf"
    print(f"分析 PDF: {pdf_path}")
    result = analyze_pdf(pdf_path)
    print("\n分析结果:\n", result)

    with open("pdf_analysis.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("\n结果已保存到 pdf_analysis.txt")