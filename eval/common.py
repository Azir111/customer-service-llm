"""
公共工具：读写 jsonl、样本归一化、加载模型、生成回复。
被 run_inference / eval_loss 复用。
"""
import json
import gc
import config


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_template_item(raw, idx):
    """把你测试集里一条样本映射成统一结构。"""
    return {
        "id": f"tmpl-{idx}",
        "bucket": "template",
        "intent": raw.get(config.FIELD_INTENT, "unknown"),
        "question": raw[config.FIELD_QUESTION],
        "reference": raw.get(config.FIELD_REFERENCE, ""),  # 仅 eval_loss 用得到
    }


# ── 模型加载 / 生成（用 Unsloth，跟训练栈一致）──────────────────────
def load_model(spec):
    """spec = {'base':..., 'adapter':...}。adapter 为 None 时加载纯基座。"""
    from unsloth import FastLanguageModel
    name = spec["adapter"] if spec["adapter"] else spec["base"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        load_in_4bit=config.LOAD_IN_4BIT,
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def free(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(model, tokenizer, question, system=None):
    import torch
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=inputs, **config.GEN)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()
