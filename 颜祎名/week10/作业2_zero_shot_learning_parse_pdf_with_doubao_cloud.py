import os
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv
import openai # 使用openai库来调用豆包的API
from PIL import Image
import io

# --- 准备工作 ---
# 建议将API Key存储在.env文件中，更安全
# .env 文件内容: DOUBAO_API_KEY="your_doubao_api_key"
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'env', '.env')
load_dotenv(dotenv_path = dotenv_path)

# 初始化豆包AI客户端
client = openai.OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3", # 豆包Ark API的Base URL
)

# --- 1. PDF 解析模块 ---
# 这个函数负责从本地PDF中提取文本和图片
def parse_pdf(pdf_path: str, output_image_folder: str = "extracted_images_doubao"):
    """
    从PDF文件中提取所有文本和图片。
    图片将被保存到本地，并返回其路径列表。
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"错误：PDF文件 '{pdf_path}' 不存在。")

    # 创建用于存放提取图片的文件夹
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    doc = fitz.open(pdf_path)
    full_text = ""
    image_paths = []

    print(f"--- 开始解析PDF: {pdf_path} ---")
    # 遍历每一页
    for page_num, page in enumerate(doc):
        # 提取文本
        full_text += page.get_text("text") + "\n"

        # 提取图片
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 将图片保存到本地文件
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_image_folder, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
            
    print(f"PDF文本提取完成。共提取 {len(image_paths)} 张图片。")
    return full_text, image_paths

# --- 2. 图片编码模块 ---
# API无法直接接收图片路径，需要将图片编码为Base64字符串
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- 3. 主逻辑 ---
if __name__ == "__main__":
    # 指定你要解析的本地PDF文件
    # 请确保在项目根目录下有一个名为 'sample.pdf' 的文件
    pdf_to_parse = "2017 Attention Is All You Need.pdf"

    try:
        # 步骤1：从PDF提取内容
        text_content, image_paths = parse_pdf(pdf_to_parse)

        # 步骤2：构建发送给豆包大模型的消息体
        messages = []
        
        # 首先，构建包含文本和图片URL的消息内容
        content_parts = [
            {
                "type": "text",
                "text": f"你是一个专业的PDF文档解析助手。请根据我提供的以下文本和图片内容，对整个文档进行总结和分析。\n\n【文档文本内容】:\n{text_content}"
            }
        ]
        
        print("\n--- 准备发送给云端豆包大模型的数据 ---")
        # 将提取出的图片编码并加入到消息中
        if image_paths:
            print(f"正在将 {len(image_paths)} 张图片编码为Base64...")
            for img_path in image_paths:
                base64_image = encode_image_to_base64(img_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high" # 建议为多模态模型指定图片细节级别
                    }
                })
        
        messages.append({"role": "user", "content": content_parts})

        # 步骤3：调用云端API
        print("\n--- 正在调用云端豆包大模型进行解析 ---")
        response = client.chat.completions.create(
            model="doubao-seed-2-0-lite-260215",  # 使用豆包的多模态模型
            messages=messages,
            stream=False # 如果不需要流式输出，可以设置为False
        )

        # 步骤4：打印模型的解析结果
        print("\n--- 云端模型解析结果 ---")
        print(response.choices[0].message.content)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生了一个未知错误: {e}")
