import os
import re
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

_bge_model = None
_clip_model = None


def get_bge_model() -> SentenceTransformer:
    global _bge_model
    if _bge_model is None:
        _bge_model = SentenceTransformer(settings.BGE_MODEL_PATH)
    return _bge_model


def get_clip_model() -> SentenceTransformer:
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(
            settings.CLIP_MODEL_PATH, trust_remote_code=True, truncate_dim=settings.CLIP_DIM
        )
    return _clip_model


def encode_text(text: str) -> tuple:
    """编码纯文本，返回 (bge_embedding, clip_embedding)"""
    bge_model = get_bge_model()
    clip_model = get_clip_model()

    try:
        bge_emb = bge_model.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        traceback.print_exc()
        bge_emb = np.zeros(settings.BGE_DIM).tolist()

    try:
        clip_emb = clip_model.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        traceback.print_exc()
        clip_emb = np.zeros(settings.CLIP_DIM).tolist()

    return bge_emb, clip_emb


def encode_image(image_path: str) -> list:
    """编码图片，返回 clip_embedding"""
    clip_model = get_clip_model()
    try:
        emb = clip_model.encode(image_path, normalize_embeddings=True).tolist()
    except Exception:
        traceback.print_exc()
        emb = np.zeros(settings.CLIP_DIM).tolist()
    return emb


def split_text2chunks(lines: list, chunk_size: int = None) -> list:
    """将文本分割成多个 chunk"""
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE

    chunks = []
    for line in lines:
        line = line.strip()
        if not line or line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if not chunks or len(chunks[-1]) > chunk_size:
            chunks.append(line)
        else:
            chunks[-1] += "\n" + line
    return chunks


def extract_image_from_markdown(markdown_text: str, markdown_dir: str) -> list:
    """从 markdown 文本中提取图片路径"""
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    images = []
    for match in pattern.finditer(markdown_text):
        img_rel = match.group(1)
        img_abs = os.path.join(markdown_dir, os.path.basename(img_rel))
        if os.path.exists(img_abs):
            images.append(img_abs)
    return images
