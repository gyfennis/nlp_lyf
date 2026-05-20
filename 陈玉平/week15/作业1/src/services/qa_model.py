import torch
from qwen_vl_utils import process_vision_info
from src.config import config


class QAModel:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            config.QWEN_VL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(config.QWEN_VL_MODEL_PATH)

    def answer(self, query: str, context_texts: list, context_images: list) -> str:
        """
        根据问题和上下文生成答案
        """
        content = []

        # 添加文本上下文
        for text in context_texts:
            content.append({"type": "text", "text": text})

        # 添加图像上下文
        for img_path in context_images:
            content.append({"type": "image", "image": f"file://{img_path}"})

        messages = [
            {
                "role": "user",
                "content": content + [{"type": "text", "text": query}]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response


qa_model = QAModel()