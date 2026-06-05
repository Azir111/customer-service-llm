# serve.py - 简易推理服务 + 性能测试（修复负延迟版）
from unsloth import FastLanguageModel
import time
import torch

# ★ 与 prepare_data / train_sft / train_dpo 逐字一致的系统提示
SYSTEM = "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"

print("加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_dpo",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
# 显式包含 <|im_end|> 作为停止符，避免 run-on
IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
EOS_IDS = [tokenizer.eos_token_id, IM_END]
print("模型加载完成！")


def chat(user_input, deterministic=False, max_new_tokens=256):
    """deterministic=True：贪心，用于性能测量/稳定输出；False：低温采样，看自然回复。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,                       # ★ 一并取 attention_mask，消除告警、结果更可靠
    )
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)

    gen = dict(max_new_tokens=max_new_tokens, attention_mask=attn,
               eos_token_id=EOS_IDS, pad_token_id=tokenizer.eos_token_id)
    if deterministic:
        gen.update(do_sample=False)
    else:
        gen.update(do_sample=True, temperature=0.3, top_p=0.9)   # ★ 0.7→0.3，降低乱跑概率

    if torch.cuda.is_available():
        torch.cuda.synchronize()                # 计时前同步
    start = time.perf_counter()                 # ★ perf_counter 比 time.time 更适合计时
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, **gen)
    if torch.cuda.is_available():
        torch.cuda.synchronize()                # 计时后同步，确保 GPU 真算完
    latency = (time.perf_counter() - start) * 1000

    new_tokens = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer, latency, len(new_tokens)


def percentile(sorted_xs, p):
    if not sorted_xs:
        return float("nan")
    k = (len(sorted_xs) - 1) * p
    lo = int(k); hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


QUESTIONS = [
    "我的订单三天了还没发货怎么办",
    "收到的东西是坏的，我要退货",
    "你们服务太差了！等了半小时没人接",
    "这个手机支持5G吗",
    "保修期是多久",
    "我要申请退款",
    "快递显示已签收但我没收到",
    "可以换货吗",
]


def spot_check():
    print("\n" + "=" * 50)
    print("🔎 效果眼检（低温采样 temperature=0.3）")
    print("=" * 50)
    for q in QUESTIONS:
        answer, _, _ = chat(q, deterministic=False)
        print(f"\n问题：{q}\n回复：{answer}\n" + "-" * 40)


def benchmark(repeats=6, warmup=8):
    """贪心解码 + 充分预热 + 丢弃每题头部样本，根治负延迟/冷启动噪声。"""
    print("\n" + "=" * 50)
    print(f"📊 性能测量（贪心 · 预热{warmup}次 · 每题×{repeats} 丢首次）")
    print("=" * 50)

    for _ in range(warmup):                     # ★ 充分预热，吃掉编译/显存分配/cudnn 选优
        chat("你好", deterministic=True, max_new_tokens=32)

    latencies, token_counts = [], []
    for q in QUESTIONS:
        per_q = []
        for r in range(repeats):
            _, latency, tokens = chat(q, deterministic=True)
            per_q.append((latency, tokens))
        # ★ 丢弃该题第一次（冷启动），其余计入
        for latency, tokens in per_q[1:]:
            latencies.append(latency)
            token_counts.append(tokens)

    # 防御：理论上同步后不会有负值，万一仍有则剔除并提示
    bad = [l for l in latencies if l <= 0]
    if bad:
        print(f"⚠ 剔除 {len(bad)} 个异常(≤0)样本")
        keep = [(l, t) for l, t in zip(latencies, token_counts) if l > 0]
        latencies, token_counts = [l for l, _ in keep], [t for _, t in keep]

    n = len(latencies)
    s = sorted(latencies)
    e2e_tok_s = sum(token_counts) / sum(l / 1000 for l in latencies)

    print(f"\n样本量:           {n}")
    print(f"平均延迟:         {sum(latencies)/n:.0f} ms")
    print(f"P50 延迟:         {percentile(s, 0.50):.0f} ms")
    if n >= 20:
        print(f"P90 延迟:         {percentile(s, 0.90):.0f} ms")
        print(f"P99 延迟:         {percentile(s, 0.99):.0f} ms")
    else:
        print("P90/P99:          样本不足 20，不输出")
    print(f"最快 / 最慢:      {min(latencies):.0f} / {max(latencies):.0f} ms")
    print(f"端到端吞吐:       {e2e_tok_s:.1f} tokens/s（含 prefill）")
    print(f"平均输出长度:     {sum(token_counts)/n:.0f} tokens")
    print("\n注：bitsandbytes 4bit 单请求，非优化推理后端；FP16/高并发见压测仓库。")


if __name__ == "__main__":
    spot_check()
    benchmark()