"""
用【测试集 loss】而不是 train loss 来验证 "epoch=3 过拟合" 的说法。

过拟合的判据是: 训练 loss 继续降、但测试 loss 开始升。
只看 train_loss 2.901→0.199 / 0.038 是无法判断过拟合的。
本脚本对若干 SFT checkpoint，在测试集参考答案上算 token 级交叉熵损失，
正确做法是只对【回复部分】的 token 计 loss（prompt 部分用 -100 mask 掉）。

把要对比的 checkpoint 填进 CKPTS（比如 epoch2 / epoch3 各存一份），跑出来比大小:
若 epoch3 的 test loss 反而比 epoch2 高 → 你"epoch=3 过拟合"的结论才算被证据支撑。

用法:  python eval/eval_loss.py
"""
import torch
import config
from common import read_jsonl, load_model, free

# ← 填你想对比的 SFT checkpoint 目录
CKPTS = {
    "sft_epoch2": "outputs/sft_lora_epoch2",
    "sft_epoch3": "outputs/sft_lora_epoch3",
}

SYSTEM = "你是一名专业的电商客服助手。"


def item_loss(model, tok, question, reference):
    """只对 reference(回复) 的 token 计 CE 损失。"""
    prompt_ids = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    ref_ids = tok(reference, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prompt_ids, ref_ids], dim=1).to(model.device)
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100  # mask 掉 prompt，只算回复
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    return out.loss.item()


def main():
    test = [r for r in read_jsonl(config.TEMPLATE_TEST_PATH)]
    # 归一化字段
    samples = [(r[config.FIELD_QUESTION], r.get(config.FIELD_REFERENCE, "")) for r in test]
    samples = [(q, a) for q, a in samples if a]  # 没参考答案的跳过
    print(f"测试集可用 {len(samples)} 条（含参考答案）")

    results = {}
    for tag, path in CKPTS.items():
        print(f"\n=== {tag} ← {path} ===")
        model, tok = load_model({"base": config.BASE_MODEL, "adapter": path})
        losses = [item_loss(model, tok, q, a) for q, a in samples]
        mean = sum(losses) / len(losses)
        results[tag] = mean
        print(f"  测试集平均 loss = {mean:.4f}")
        free(model)

    print("\n──── 结论 ────")
    for tag, v in results.items():
        print(f"  {tag:14s} test_loss = {v:.4f}")
    if len(results) == 2:
        a, b = list(results.items())
        verdict = ("更晚的 epoch 测试 loss 更高 → 支持过拟合说法"
                   if list(results.values())[1] > list(results.values())[0]
                   else "更晚的 epoch 测试 loss 没升 → 当前不能断言过拟合，理由需重写")
        print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
