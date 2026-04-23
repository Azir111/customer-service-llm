# serve.py - 简易推理服务 + 性能测试
from unsloth import FastLanguageModel
import time
import torch

print("加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_dpo",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
print("模型加载完成！")

def chat(user_input):
    messages = [
        {"role": "system", "content": "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"},
        {"role": "user", "content": user_input}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - start) * 1000

    new_tokens = outputs[0][inputs.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    token_count = len(new_tokens)

    return answer, latency, token_count


def benchmark():
    questions = [
        "我的订单三天了还没发货怎么办",
        "收到的东西是坏的，我要退货",
        "你们服务太差了！等了半小时没人接",
        "这个手机支持5G吗",
        "保修期是多久",
        "我要申请退款",
        "快递显示已签收但我没收到",
        "可以换货吗",
    ]

    print("\n" + "=" * 50)
    print("📊 效果 + 性能测试")
    print("=" * 50)

    latencies = []
    token_counts = []

    # 预热一次（第一次推理会慢）
    chat("你好")

    for q in questions:
        answer, latency, tokens = chat(q)
        latencies.append(latency)
        token_counts.append(tokens)
        print(f"\n问题：{q}")
        print(f"回复：{answer}")
        print(f"延迟：{latency:.0f}ms | 生成tokens：{tokens}")
        print("-" * 40)

    # 汇总
    latencies_sorted = sorted(latencies)
    avg_tokens_per_sec = sum(token_counts) / sum(l/1000 for l in latencies)

    print("\n" + "=" * 50)
    print("📈 性能汇总")
    print("=" * 50)
    print(f"平均延迟:       {sum(latencies)/len(latencies):.0f} ms")
    print(f"P90 延迟:       {latencies_sorted[int(len(latencies)*0.9)]:.0f} ms")
    print(f"最快响应:       {min(latencies):.0f} ms")
    print(f"最慢响应:       {max(latencies):.0f} ms")
    print(f"平均生成速度:   {avg_tokens_per_sec:.1f} tokens/s")
    print(f"平均输出长度:   {sum(token_counts)/len(token_counts):.0f} tokens")

benchmark()
