# train_dpo.py（修改版）
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import json

PatchDPOTrainer()  # unsloth 的 DPO 补丁，提速

# ★★ 全流程唯一的系统提示：必须和 SFT 训练、eval/run_inference.py、serve.py 逐字一致 ★★
#    这就是 prepare_data.py 里写进 instruction 字段的那句。改这里就要四处一起改，
#    否则训练用一种 prompt、推理/评测用另一种，模型表现会被人为压低。
SYSTEM = "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"

# 1) 加载 SFT 后的 LoRA
#    换数据重训后，这里要指向【新的】SFT 输出目录（旧 adapter 是在泄漏数据上训的，作废）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_lora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_training(model)  # 切换到训练模式

# 2) 读取偏好数据（路径按你实际位置改；DPO 数据是 chosen/rejected 偏好对，和 SFT 的 train/test 不是一回事）
with open("data/data/dpo_data.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# 3) ★ 关键修改：把 prompt 用 chat template 拼成与 SFT/eval/serve 完全一致的格式
#    DPOTrainer 需要每条样本含 {prompt, chosen, rejected}：
#      - prompt：系统提示 + 用户问题，经 chat template，并带 add_generation_prompt（结尾是 assistant 起始符）
#      - chosen / rejected：只放【回复正文】，不要带 <|im_start|>assistant 之类角色标签（模板已经加好了）
#    下面字段名按你 dpo_data.json 的真实结构改（常见是 question / input / prompt + chosen + rejected）。
def to_dpo(item):
    question = item.get("question") or item.get("input") or item["prompt"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "chosen": item["chosen"], "rejected": item["rejected"]}

dataset = Dataset.from_list([to_dpo(x) for x in raw])
print(f"DPO 偏好样本: {len(dataset)} 条")
print("prompt 样例:\n", dataset[0]["prompt"])  # 跑前肉眼核对一下格式对不对

# 4) DPO 训练（小数据下注意总步数别太少，见文末说明）
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