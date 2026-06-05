# diagnose.py —— 分离"采样的锅"和"权重的锅"，并检查是否正常停止
from unsloth import FastLanguageModel
import torch

model, tok = FastLanguageModel.from_pretrained(
    model_name="output/customer_service_dpo", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

SYSTEM = "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"
im_end = tok.convert_tokens_to_ids("<|im_end|>")

print("eos_token:", tok.eos_token, "| eos_id:", tok.eos_token_id, "| <|im_end|> id:", im_end)
print("generation_config.eos_token_id:", getattr(model.generation_config, "eos_token_id", None))
print("=" * 60)

QS = ["我要申请退款", "可以换货吗", "保修期是多久", "我的订单三天了还没发货怎么办"]

def run(q, **gen):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=200, **gen)
    new = out[0][ids.shape[-1]:]
    clean = tok.decode(new, skip_special_tokens=True)
    raw = tok.decode(new, skip_special_tokens=False)   # 看是否吐了 <|im_end|>
    stopped = "<|im_end|>" in raw
    return clean, stopped, raw[-30:]

for q in QS:
    print("问:", q)
    c, stopped, tail = run(q, do_sample=False, eos_token_id=[tok.eos_token_id, im_end])
    print("  [贪心+显式停止] 是否吐<|im_end|>:", stopped)
    print("  回复:", c)
    print("  原始结尾:", repr(tail))
    print("-" * 60)