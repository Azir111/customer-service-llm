"""
汇总 judge 结果，按 bucket(template/hard) 和 intent 拆开统计，打印并写 eval/report.md。

用法:  python eval/report.py
产物:  eval/report.md
"""
from collections import defaultdict
import config
from common import read_jsonl

DIMS = ["correctness", "helpfulness", "tone", "safety"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pointwise_tables():
    rows = read_jsonl("eval/outputs/pointwise.jsonl")
    # 按 (model, bucket) 聚合每个维度均分
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for d in DIMS:
            if d in r:
                agg[(r["model"], r["bucket"])][d].append(r[d])
    lines = ["### 逐条打分（1~5，均分）\n",
             "| 模型 | 数据桶 | 正确性 | 有用性 | 语气 | 安全 | 总均分 |",
             "|------|--------|--------|--------|------|------|--------|"]
    for (model, bucket), dim_scores in sorted(agg.items()):
        vals = [mean(dim_scores[d]) for d in DIMS]
        overall = mean([v for v in vals])
        lines.append(f"| {model} | {bucket} | " +
                     " | ".join(f"{v:.2f}" for v in vals) + f" | {overall:.2f} |")
    return "\n".join(lines)


def pairwise_tables():
    rows = read_jsonl("eval/outputs/pairwise.jsonl")
    out = ["\n### 成对胜负（headline 指标）\n"]
    # 按 compare + bucket 统计胜率
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [win, lose, tie]
    for r in rows:
        win_tag = r["compare"].split("_vs_")[0]
        cell = agg[r["compare"]][r["bucket"]]
        if r["winner"] == win_tag:
            cell[0] += 1
        elif r["winner"] == "tie":
            cell[2] += 1
        else:
            cell[1] += 1
    for compare, buckets in agg.items():
        win_tag, lose_tag = compare.split("_vs_")
        out.append(f"**{win_tag} vs {lose_tag}** （{win_tag} 的胜率 / 平 / 负）\n")
        out.append("| 数据桶 | 胜 | 平 | 负 | 胜率(不含平) |")
        out.append("|--------|----|----|----|--------------|")
        for bucket, (w, l, t) in sorted(buckets.items()):
            decisive = w + l
            wr = f"{w/decisive*100:.0f}%" if decisive else "—"
            out.append(f"| {bucket} | {w} | {t} | {l} | {wr} |")
        out.append("")
    return "\n".join(out)


def main():
    parts = ["# 评测报告\n",
             f"被测模型: {', '.join(config.MODELS.keys())}  ",
             f"裁判: {config.JUDGE_MODEL} @ {config.JUDGE_BASE_URL}\n",
             "> template = 同分布模板测试集（衡量模板内拟合）；"
             "hard = 手写非模板探针（衡量真正泛化与越界处理）。\n",
             pointwise_tables(),
             pairwise_tables(),
             "## 怎么读这份报告\n",
             "- 看 **sft vs base** 的胜率：>50% 才说明微调确实带来提升，否则白练。\n",
             "- 看 **dpo vs sft** 的胜率：这才是 DPO 真正有没有用，"
             "比训练集 rewards/margins 可信得多。\n",
             "- 重点对比 template 桶 vs hard 桶：若 hard 桶分数明显掉，"
             "说明只是过拟合了模板，泛化差——正是 README 该如实标注的边界。\n",
             "- safety 维度低 = 模型会乱承诺退款/折扣或不处理越权请求，"
             "部署风险，比分数低更值得优先修。\n"]
    report = "\n".join(parts)
    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("\n→ 已写入 eval/report.md")


if __name__ == "__main__":
    main()
