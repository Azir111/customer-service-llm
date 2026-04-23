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
dpo_data = [
    {
        "prompt": "用户说：我要投诉你们快递太慢了！",
        "chosen": "非常抱歉给您带来不好的体验！我完全理解您的心情，等待确实让人着急。请告诉我您的订单号，我立即为您优先跟进物流，并申请相应补偿。",
        "rejected": "快递速度由物流公司决定，我们无法直接控制，请您耐心等待。"
    },
    {
        "prompt": "用户说：收到的东西坏了",
        "chosen": "非常抱歉！这是我们的失误，给您带来了不便。请拍几张照片发给我，我立即为您安排免费换货，运费由我们承担，不需要您操心任何费用。",
        "rejected": "您好，请问是什么坏了？我们需要核实情况。"
    },
    {
        "prompt": "用户说：等了半小时客服还没接",
        "chosen": "非常抱歉让您久等了！这是我们服务的不足，深感歉意。我现在立即为您处理，并会将这个问题上报给团队，改善我们的响应速度。您遇到什么问题，请告诉我。",
        "rejected": "您好，我们客服人员较忙，请继续等待。"
    },
   
]

dataset = Dataset.from_list(dpo_data)

# DPO 训练
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,    # unsloth 不需要单独的参考模型
    args = DPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs = 1,
        learning_rate = 5e-5,
        beta = 0.1,          # DPO 核心超参
        output_dir = "output/dpo_checkpoints",
        report_to = "none",
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