import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


IMAGE_PATH = r"/Users/arvin/Desktop/Week10/homework1/dog.png"

MODEL_DIR = r"/Users/arvin/Desktop/AI/models/chinese-clip-vit-base-patch16"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChineseCLIPModel.from_pretrained(MODEL_DIR).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_DIR)

    image = Image.open(IMAGE_PATH).convert("RGB")

    candidate_texts = [
        "一只小狗",
        "一只猫",
        "一只鸟",
        "一辆汽车",
        "一碗食物",
    ]

    inputs = processor(
        text=candidate_texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (1, num_texts)
        probs = logits_per_image.softmax(dim=-1).squeeze(0)  # shape: (num_texts,)

    top_idx = int(torch.argmax(probs).item())

    print(f"device: {device}")
    print(f"image: {IMAGE_PATH}")
    print("----- zero-shot classification -----")
    for text, p in zip(candidate_texts, probs.tolist()):
        print(f"{p:8.4f}  {text}")
    print("-----------------------------------")
    print(f"Top-1: {candidate_texts[top_idx]}")


if __name__ == "__main__":
    main()

