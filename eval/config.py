"""
评测配置 —— 已对齐你的真实目录 / DeepSeek 裁判。跑前只需 export 一个 API key。
"""
import os

# ── 1. 模型：base 基线 + 你训出的 SFT / DPO adapter ──────────────────
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
MODELS = {
    "base": {"base": BASE_MODEL, "adapter": None},
    "sft":  {"base": BASE_MODEL, "adapter": "output/customer_service_lora"},   # ← 你的 SFT 输出
    "dpo":  {"base": BASE_MODEL, "adapter": "output/customer_service_dpo"},    # ← 你的 DPO 输出
}
PAIRWISE = [
    ("sft", "base"),   # 微调有没有用
    ("dpo", "sft"),    # DPO 在 SFT 之上有没有提升
]

# ── 2. 评测集：你的测试集 + 非模板探针 ──────────────────────────────
TEMPLATE_TEST_PATH = "data/data/test.jsonl"   # ← 对齐你的真实路径
HARD_CASES_PATH    = "eval/hard_cases.jsonl"
EVAL_SET_PATH      = "eval/outputs/eval_set.jsonl"

# 你测试集的字段名（你的问题在 input，不是 instruction）
FIELD_QUESTION  = "input"
FIELD_REFERENCE = "output"
FIELD_INTENT    = "intent"

# ── 3. 生成参数（评测用贪心，可复现）──────────────────────────────
GEN = dict(max_new_tokens=256, do_sample=False, temperature=0.0, top_p=1.0)
LOAD_IN_4BIT = True

# ── 4. 裁判：DeepSeek（OpenAI 兼容端点）──────────────────────────────
#    key 从环境变量读取，不写进代码： export JUDGE_API_KEY=sk-xxx
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com")
JUDGE_MODEL    = os.environ.get("JUDGE_MODEL", "deepseek-chat")
JUDGE_API_KEY  = os.environ.get("JUDGE_API_KEY", "")
JUDGE_SWAP     = True   # A/B 双向各判一次，抵消位置偏差