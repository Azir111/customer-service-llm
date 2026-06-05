"""
拼装统一评测集 = 你的 200 条模板测试集 + 手写非模板探针。
产物 eval_set.jsonl 每条带 bucket(template/hard) 和 intent，下游脚本按这两维拆开统计。

用法:  python eval/build_eval_set.py
"""
import os
import config
from common import read_jsonl, write_jsonl, normalize_template_item


def main():
    items = []

    # 1) 模板测试集（同分布，衡量模板内拟合）
    if os.path.exists(config.TEMPLATE_TEST_PATH):
        raw = read_jsonl(config.TEMPLATE_TEST_PATH)
        items += [normalize_template_item(r, i) for i, r in enumerate(raw)]
        print(f"模板测试集: {len(raw)} 条 ← {config.TEMPLATE_TEST_PATH}")
    else:
        print(f"⚠ 未找到模板测试集 {config.TEMPLATE_TEST_PATH}，本次只用 hard 集。"
              f"请在 config.py 改 TEMPLATE_TEST_PATH 和字段映射。")

    # 2) 手写非模板探针（衡量真正的泛化 / 越界处理）
    hard = read_jsonl(config.HARD_CASES_PATH)
    for h in hard:
        h.setdefault("reference", "")  # hard 集无参考答案，eval_loss 会自动跳过
    items += hard
    print(f"非模板探针: {len(hard)} 条 ← {config.HARD_CASES_PATH}")

    os.makedirs(os.path.dirname(config.EVAL_SET_PATH), exist_ok=True)
    write_jsonl(config.EVAL_SET_PATH, items)
    print(f"\n合计 {len(items)} 条 → {config.EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
