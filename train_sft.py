# train_sft.py（修改版）
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# ★★ 全流程唯一的系统提示：必须和 prepare_data.py(instruction)、train_dpo.py(SYSTEM)、serve.py 逐字一致 ★★
SYSTEM = "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"

# ===== 1. 加载模型（自动量化到4bit，显存友好）=====
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",  # 显存8GB用这个
    # model_name = "unsloth/Qwen2.5-3B-Instruct",  # 如果跑得动可以用3B
    max_seq_length = 2048,
    dtype = None,           # 自动选择
    load_in_4bit = True,    # 4bit量化，显存减半
)

# ===== 2. 添加 LoRA 适配器 =====
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                     # LoRA rank
    target_modules = [          # 作用于哪些层（Qwen的层名）
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 省显存
    random_state = 42,
)

# ===== 3. 准备数据 =====
def format_prompt(example):
    """把数据格式化成模型输入（ChatML，与 DPO/serve 保持一致）"""
    text = f"""<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    return {"text": text}

# ★ 新版数据是 JSONL（每行一个对象），路径在 data/data/，不能再用 json.load
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

train_data = read_jsonl("data/data/train.jsonl")
print(f"训练样本: {len(train_data)} 条")

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_prompt)

# ===== 4. 训练配置 =====
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # 等效batch=8
        warmup_steps = 50,
        num_train_epochs = 3,              # ★ 数据换成干净多样的了，旧"epoch=2防过拟合"的理由已失效；
                                           #   先训 3 个 epoch、每轮存一份，之后用测试集 loss 挑最优 epoch
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        save_strategy = "epoch",           # ★ 每个 epoch 存一份 checkpoint（替代原 save_steps=200），
                                           #   方便日后用 eval_loss.py 在测试集上挑 epoch
        output_dir = "output/sft_checkpoints",
        optim = "adamw_8bit",              # 8bit优化器，省显存
        report_to = "none",
    ),
)

# ===== 5. 开始训练 =====
print("开始训练...")
print(f"显存使用: {torch.cuda.memory_allocated()/1e9:.1f} GB")

trainer_stats = trainer.train()

print(f"训练完成！用时: {trainer_stats.metrics['train_runtime']/60:.1f} 分钟")

# ===== 6. 保存模型 =====
model.save_pretrained("output/customer_service_lora")
tokenizer.save_pretrained("output/customer_service_lora")
print("模型已保存到 output/customer_service_lora")