# !pip install -U accelerate chromadb sentence-transformers faiss-cpu bitsandbytes
# !pip install "git+https://github.com/huggingface/transformers"
from config import MODEL_ID, DEVICE, DTYPE
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =========================
class Qwen3VLAgent:
    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, dtype=DTYPE):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)
    def chat(self, messages, max_new_tokens=128):
        # Prepare input
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text