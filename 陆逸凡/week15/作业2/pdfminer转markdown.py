import os
from pdfminer.high_level import extract_text

def pdf_to_markdown(pdf_path):
    """
    将 PDF 文件转换为 Markdown 文本
    """
    raw_text = extract_text(pdf_path)
    # 直接返回文本
    return raw_text

# 获取当前文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))
print(f"当前文件夹: {current_folder}")

# PDF文件路径
filename = "2509-MinerU2.5.pdf"
pdf_path = os.path.join(current_folder, filename)

# 检查PDF是否存在
if not os.path.exists(pdf_path):
    print(f"错误：找不到PDF文件 {pdf_path}")
    print(f"请确认当前文件夹下有以下文件：")
    for file in os.listdir(current_folder):
        print(f"  - {file}")
else:
    # 转换PDF
    md_content = pdf_to_markdown(pdf_path)
    
    # 保存到当前文件夹
    output_path = os.path.join(current_folder, "pdfminer的输出.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"✅ Markdown文件已保存到: {output_path}")
    
    # 验证文件是否真的创建了
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ 文件创建成功，大小: {file_size} 字节")
    else:
        print("❌ 文件创建失败")