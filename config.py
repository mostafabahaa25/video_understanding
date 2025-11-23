import torch
# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32