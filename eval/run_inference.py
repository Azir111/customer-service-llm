"""
让 base / sft / dpo 三个模型分别在评测集上生成回复。
8GB 卡放不下三个模型，所以逐个加载→生成→释放，串行跑。

用法:  python eval/run_inference.py
产物:  eval/outputs/answers_{base,sft,dpo}.jsonl
"""
import os
import config
from common import read_jsonl, write_jsonl, load_model, generate, free

SYSTEM = "你是一名专业的电商客服助手，回答要准确、礼貌、有共情，"\
         "不编造不存在的政策或商品信息，遇到超出权限或需核验身份的请求应引导走正规流程。"


def main():
    eval_set = read_jsonl(config.EVAL_SET_PATH)
    print(f"评测集 {len(eval_set)} 条")

    for tag, spec in config.MODELS.items():
        print(f"\n=== 加载模型 [{tag}] ===")
        model, tok = load_model(spec)
        rows = []
        for i, item in enumerate(eval_set):
            ans = generate(model, tok, item["question"], system=SYSTEM)
            rows.append({
                "id": item["id"], "intent": item["intent"],
                "bucket": item["bucket"], "question": item["question"],
                "answer": ans,
            })
            if (i + 1) % 20 == 0:
                print(f"  [{tag}] {i+1}/{len(eval_set)}")
        out = f"eval/outputs/answers_{tag}.jsonl"
        write_jsonl(out, rows)
        print(f"  [{tag}] 写入 {out}")
        free(model)

    print("\n全部模型推理完成。下一步: python eval/judge.py")


if __name__ == "__main__":
    main()
