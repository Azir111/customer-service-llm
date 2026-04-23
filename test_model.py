# test_model.py
from unsloth import FastLanguageModel

# 加载训练好的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "output/customer_service_lora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)  # 切换到推理模式，速度更快

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
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    # 只取新生成的部分
    new_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# 测试几个典型客服场景
test_questions = [
    "我的订单三天了还没发货怎么办",
    "收到的东西是坏的，我要退货",
    "你们服务太差了！等了半小时没人接",
    "这个手机支持5G吗",
    "保修期是多久",
]

print("=" * 50)
for q in test_questions:
    print(f"\n用户：{q}")
    print(f"客服：{chat(q)}")
    print("-" * 40)