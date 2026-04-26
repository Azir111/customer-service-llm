# train_dpo.py
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import json

PatchDPOTrainer()  # unsloth 的 DPO 补丁，提速

# 加载 SFT 后的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_lora",  # 加载已有LoRA
    max_seq_length = 2048,
    load_in_4bit = True,
)
# 直接用，不需要再 get_peft_model，模型已经有适配器了
FastLanguageModel.for_training(model)  # 切换到训练模式

# DPO 数据（好回复 vs 差回复）


with open("data/dpo_data.json", "r", encoding="utf-8") as f:
    dpo_data = json.load(f)

dataset = Dataset.from_list(dpo_data)

# DPO 训练
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,    # unsloth 不需要单独的参考模型
    args = DPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        num_train_epochs = 2,
        learning_rate = 5e-5,
        beta = 0.1,          # DPO 核心超参
        output_dir = "output/dpo_checkpoints",
        report_to = "none",
        logging_steps = 1,
    ),
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_length = 1024,
    max_prompt_length = 512,
)

dpo_trainer.train()
model.save_pretrained("output/customer_service_dpo")
tokenizer.save_pretrained("output/customer_service_dpo")
print("DPO 训练完成！")