import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2"
print(f"Loading {model_name} in bf16 with Flash Attention 2...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("Model loaded successfully with Flash Attention 2!")
    print(f"Model device: {model.device}")
except Exception as e:
    print(f"Error loading model: {e}")
