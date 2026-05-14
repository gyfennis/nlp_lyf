from openai import OpenAI
import os
import base64
import io
import pypdfium2 as pdfium

# ===================== 配置区域 =====================
# 你的 API Key
api_key = "sk-054668d7012c4e26bc560e1d8d51198c"
# 你的 PDF 路径
pdf_path = "./作业2_01_应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"
# ====================================================

# 初始化客户端（完全沿用你给的格式！）
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# PDF 第一页 → 转 base64（核心功能）
def pdf_to_base64(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(0)
    pil_image = page.render().to_pil()
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# 转换 PDF
image_base64 = pdf_to_base64(pdf_path)

# 拼接成 Qwen-VL 支持的 base64 图片格式
image_url = f"data:image/png;base64,{image_base64}"

# ===================== 完全沿用你的流式输出格式 =====================
reasoning_content = ""
answer_content = ""
is_answering = False

completion = client.chat.completions.create(
    model="qvq-max",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "详细解析这一页PDF的内容"},
            ],
        }
    ],
    stream=True,
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta

    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
        print(delta.reasoning_content, end='', flush=True)
        reasoning_content += delta.reasoning_content
    else:
        if delta.content and not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True
        if delta.content:
            print(delta.content, end='', flush=True)
            answer_content += delta.content