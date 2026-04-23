# export_model.py
from unsloth import FastLanguageModel

print("加载 DPO 训练后的模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_dpo",
    max_seq_length = 2048,
    load_in_4bit = True,
)

print("导出为 GGUF 格式（约需 3~5 分钟）...")
model.save_pretrained_gguf(
    "output/customer_service_gguf",
    tokenizer,
    quantization_method = "q4_k_m"
)

print("导出完成！文件在 output/customer_service_gguf/")