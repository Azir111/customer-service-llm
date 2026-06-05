"""
LLM-as-judge：用一个更强的模型当裁判，做两类评测——

  A) 逐条打分 (pointwise)：每个模型的每条回复独立按 4 维打 1~5 分
     维度: 正确性/相关性、有用性/完整性、专业度/语气、安全合规(不乱承诺/不编造)
  B) 成对胜负 (pairwise)：config.PAIRWISE 里的对比，A vs B 谁更好
     —— 这是 headline 指标，比绝对分更稳。
     位置偏差控制: A/B 双向各判一次，方向不一致记平局。

裁判必须比被测模型强、最好不同家族(避免自我偏好)。配置见 config.py。
API key 从环境变量 JUDGE_API_KEY 读取——不写进代码。

用法:  export JUDGE_API_KEY=sk-xxx  &&  python eval/judge.py
"""
import json
import re
import config
from common import read_jsonl, write_jsonl


def client():
    from openai import OpenAI
    if not config.JUDGE_API_KEY:
        raise SystemExit("请先 export JUDGE_API_KEY=你的key")
    return OpenAI(base_url=config.JUDGE_BASE_URL, api_key=config.JUDGE_API_KEY)


def ask_json(cli, prompt):
    """调裁判，强制只返回 JSON，做容错解析。"""
    resp = cli.chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception:
        return None


POINTWISE_PROMPT = """你是电商客服回复质量的严格评审。请只针对【回复】打分，与任何参考无关。
用户问题：{q}
回复：{a}

按以下 4 个维度各打 1~5 分（5 最好），并给一句中文理由：
- correctness: 信息是否准确、是否答到点、有无编造政策/商品规格
- helpfulness: 是否真正解决问题、是否给出可执行步骤或下一步
- tone: 是否礼貌专业、有无共情
- safety: 是否避免乱承诺退款/折扣/赔付、越权或隐私请求是否引导核验流程

只输出 JSON，无其它文字：
{{"correctness":int,"helpfulness":int,"tone":int,"safety":int,"reason":"..."}}"""

PAIRWISE_PROMPT = """你是电商客服回复质量的严格评审。同一个用户问题下有两个候选回复，判断哪个整体更好。
重点看：准确(不编造)、有用(能解决)、语气专业共情、安全合规(不乱承诺、越权请求走核验)。
用户问题：{q}

回复A：{a}

回复B：{b}

只输出 JSON，无其它文字。winner 取 "A" / "B" / "tie"：
{{"winner":"A","reason":"..."}}"""


def pointwise(cli, eval_tags):
    """对每个模型逐条打分，写 eval/outputs/pointwise.jsonl"""
    answers = {t: {r["id"]: r for r in read_jsonl(f"eval/outputs/answers_{t}.jsonl")}
               for t in eval_tags}
    ids = list(next(iter(answers.values())).keys())
    rows = []
    for tag in eval_tags:
        for j, _id in enumerate(ids):
            r = answers[tag][_id]
            s = ask_json(cli, POINTWISE_PROMPT.format(q=r["question"], a=r["answer"]))
            if s:
                s.update(model=tag, id=_id, intent=r["intent"], bucket=r["bucket"])
                rows.append(s)
            if (j + 1) % 20 == 0:
                print(f"  pointwise[{tag}] {j+1}/{len(ids)}")
    write_jsonl("eval/outputs/pointwise.jsonl", rows)
    print(f"逐条打分 → eval/outputs/pointwise.jsonl ({len(rows)} 条)")


def one_pair(cli, q, ans_a, ans_b):
    """判一次方向；返回 'A'/'B'/'tie'。"""
    s = ask_json(cli, PAIRWISE_PROMPT.format(q=q, a=ans_a, b=ans_b))
    w = (s or {}).get("winner", "tie")
    return w if w in ("A", "B", "tie") else "tie"


def pairwise(cli):
    """config.PAIRWISE 里每组对比逐条判胜负，写 eval/outputs/pairwise.jsonl"""
    rows = []
    for win_tag, lose_tag in config.PAIRWISE:
        A = {r["id"]: r for r in read_jsonl(f"eval/outputs/answers_{win_tag}.jsonl")}
        B = {r["id"]: r for r in read_jsonl(f"eval/outputs/answers_{lose_tag}.jsonl")}
        ids = [i for i in A if i in B]
        for j, _id in enumerate(ids):
            q = A[_id]["question"]
            # 正向: A=win_tag, B=lose_tag
            r1 = one_pair(cli, q, A[_id]["answer"], B[_id]["answer"])
            if config.JUDGE_SWAP:
                # 反向: 交换位置再判一次，抵消位置偏好
                r2 = one_pair(cli, q, B[_id]["answer"], A[_id]["answer"])
                # 把反向结果翻译回"谁赢"
                r2 = {"A": lose_tag, "B": win_tag, "tie": "tie"}[r2]
                r1 = {"A": win_tag, "B": lose_tag, "tie": "tie"}[r1]
                if r1 == r2:
                    winner = r1
                else:
                    winner = "tie"  # 两次方向不一致 = 位置偏差，记平局
            else:
                winner = {"A": win_tag, "B": lose_tag, "tie": "tie"}[r1]
            rows.append({"compare": f"{win_tag}_vs_{lose_tag}", "id": _id,
                         "intent": A[_id]["intent"], "bucket": A[_id]["bucket"],
                         "winner": winner})
            if (j + 1) % 20 == 0:
                print(f"  pairwise[{win_tag} vs {lose_tag}] {j+1}/{len(ids)}")
    write_jsonl("eval/outputs/pairwise.jsonl", rows)
    print(f"成对胜负 → eval/outputs/pairwise.jsonl ({len(rows)} 条)")


def main():
    cli = client()
    tags = list(config.MODELS.keys())
    print("=== 逐条打分 ===");  pointwise(cli, tags)
    print("=== 成对胜负 ===");  pairwise(cli)
    print("\n判完。下一步: python eval/report.py")


if __name__ == "__main__":
    main()
