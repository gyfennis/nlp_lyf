
import base64
from io import BytesIO

PDF_PATH = "/Users/arvin/Desktop/Week10/homework2/应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"

DASHSCOPE_API_KEY = "sk-9f06aac1b31541958699954fe1ca8432"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

QWEN_VL_MODEL = "qwen-vl-plus"

PROMPT = "请解析这页 PDF 的内容：尽可能提取标题、段落、表格/公式（如有），并用结构化要点输出。"

DPI = 200
MAX_TOKENS = 2048

try:
    import fitz
except Exception as e:
    raise SystemExit(
        "Missing dependency: pymupdf\n"
        "Install with: pip install pymupdf\n"
        f"Original error: {e}"
    )

try:
    from PIL import Image
except Exception as e:
    raise SystemExit(
        "Missing dependency: pillow\n"
        "Install with: pip install pillow\n"
        f"Original error: {e}"
    )

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "Missing dependency: openai\n"
        "Install with: pip install openai\n"
        f"Original error: {e}"
    )


def pdf_first_page_to_png_base64(pdf_path: str, *, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    if doc.page_count < 1:
        raise ValueError("PDF has no pages.")

    page = doc.load_page(0)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_with_qwen_vl(
    *,
    api_key: str,
    png_b64: str,
    prompt: str,
    model: str,
    base_url: str,
    max_tokens: int,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> int:
    if not PDF_PATH or PDF_PATH == "/absolute/path/to/your.pdf":
        raise SystemExit("Please set PDF_PATH in the CONFIG section at the top of this file.")

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit(
            "Please set DASHSCOPE_API_KEY in the CONFIG section at the top of this file."
        )

    png_b64 = pdf_first_page_to_png_base64(PDF_PATH, dpi=DPI)
    text = parse_with_qwen_vl(
        api_key=DASHSCOPE_API_KEY,
        png_b64=png_b64,
        prompt=PROMPT,
        model=QWEN_VL_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=MAX_TOKENS,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

