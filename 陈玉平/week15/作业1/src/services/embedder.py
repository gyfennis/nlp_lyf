import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import clip
from src.config import config


class TextEmbedder:
    def __init__(self):
        self.model = AutoModel.from_pretrained(config.BGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.BGE_MODEL_NAME)
        self.model.to(config.QWEN_VL_DEVICE)
        self.model.eval()

    def encode(self, texts: list) -> list:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(config.QWEN_VL_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings.tolist()


class ImageEmbedder:
    def __init__(self):
        self.device = config.QWEN_VL_DEVICE
        self.model, self.preprocess = clip.load(config.CLIP_MODEL_NAME, device=self.device)

    def encode(self, images: list) -> list:
        images_tensor = []
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            images_tensor.append(image)

        images_tensor = torch.cat(images_tensor, dim=0)

        with torch.no_grad():
            features = self.model.encode_image(images_tensor)

        return features.cpu().numpy().tolist()


text_embedder = TextEmbedder()
image_embedder = ImageEmbedder()