"""
prepare_dpo_data.py

生成 DPO 偏好数据（chosen / rejected 对）。和 SFT 的 train/test 是两套数据。

设计要点：
- chosen   = 好回复：准确、有共情、给可执行步骤、该要信息就要（如订单号）、不乱承诺。
- rejected = 差回复，但都是【真实失败模式】，这样 DPO 才有意义可学，对应评测的 safety 维度：
    1) 乱承诺      —— "马上给您全额退款/打五折"（越权、不核实政策）
    2) 无共情/敷衍  —— "你自己去查" "过保了不管"
    3) 编造信息    —— 瞎报规格、虚构"明天一定到"
    4) 答非所问    —— 不解决问题、不给下一步
- 槽位填充保证多样性；并和 test.jsonl 去重，避免偏好数据撞上测试集。

输出 JSON 数组到 data/data/dpo_data.json，字段 {question, chosen, rejected, intent}，
对齐修改版 train_dpo.py 的读取逻辑。
"""
import json
import os
import random

random.seed(7)

OUT_PATH  = "data/data/dpo_data.json"
TEST_PATH = "data/data/test.jsonl"   # 用于去重；不存在则跳过
N_PER_INTENT = 24                    # 每类意图目标条数（5 类 ≈ 120 条）

ORDER_NO = lambda: "".join(random.choices("0123456789", k=random.choice([12, 15, 18])))
PRODUCTS = ["蓝牙耳机", "保温杯", "运动鞋", "充电宝", "护肤套装", "机械键盘",
            "羽绒服", "猫粮", "台灯", "行李箱", "电饭煲", "瑜伽垫"]
DAYS     = ["昨天", "前天", "三天前", "上周一", "5 天前", "一周前"]


def logistics():
    p, no, d = random.choice(PRODUCTS), ORDER_NO(), random.choice(DAYS)
    q = f"我{d}买的{p}到哪了？订单号{no}"
    chosen = (f"您好！已为您记录订单{no}，我马上为您查询{p}的实时物流。"
              f"通常标准配送 3-5 个工作日，若有延误我会主动为您跟进，请稍候。")
    rejected = random.choice([
        f"您的{p}明天肯定能到，放心吧。",                 # 编造确定性
        "这个我也不知道，您自己去快递官网查吧。",          # 无共情/敷衍
        "好的。",                                       # 答非所问
    ])
    return q, chosen, rejected


def return_refund():
    p, no = random.choice(PRODUCTS), ORDER_NO()
    cond = random.choice(["用了几天觉得不合适", "买错型号了", "不想要了"])
    q = f"订单{no}的{p}我想退货，{cond}"
    chosen = (f"您好！很抱歉{p}没能满足您的需求。是否可退要看是否在 7 天无理由期内及商品状态，"
              f"我先帮您核实订单{no}的时效，符合条件就指导您申请退货，不符合也会给您其他方案。")
    rejected = random.choice([
        f"没问题！我现在就给订单{no}全额退款，运费也免了。",   # 乱承诺/越权（safety 失败）
        "用过了一般不能退，您自己看着办吧。",                # 无共情且不准确
        "退货请走流程。",                                  # 不给可执行步骤
    ])
    return q, chosen, rejected


def consult():
    p = random.choice(PRODUCTS)
    asp = random.choice(["支持几瓦快充", "材质是什么", "尺码怎么选", "保质期多久"])
    q = f"这个{p}{asp}？"
    chosen = (f"您好！关于{p}的「{asp}」，详情页有完整参数；为避免给您错误信息，"
              f"具体规格请以商品详情页和官方参数为准，我也可以帮您逐项核对。")
    rejected = random.choice([
        f"{p}支持 200W 超级快充，全球最快。",            # 编造规格
        "这个详情页都有，自己看。",                       # 无共情/敷衍
        "应该可以吧。",                                  # 不确定还硬答
    ])
    return q, chosen, rejected


def complaint():
    issue = random.choice([
        "客服半天没人理", "APP 老闪退", "等了很久没人接电话", "发货太慢",
        "客服态度敷衍", "页面价格和结算不一致", "优惠券用不了", "包装破破烂烂",
        "退款迟迟不到账", "下单后被偷偷砍单",
    ])
    q = random.choice([
        f"你们{issue}，太差了！",
        f"{issue}，我要投诉",
        f"{issue}，再不解决我就投诉了",
        f"对你们很失望，{issue}",
    ])
    chosen = (f"非常抱歉给您带来了不好的体验，「{issue}」确实是我们的问题，向您致歉。"
              f"您的情况我现在优先处理，同时反馈给相关团队改进，避免再次发生。")
    rejected = random.choice([
        "这不是我们的问题，是您自己操作的问题。",          # 推责、无共情
        "投诉请打官方电话。",                            # 踢皮球
        "知道了。",                                     # 敷衍
    ])
    return q, chosen, rejected


def aftersale():
    p, no = random.choice(PRODUCTS), ORDER_NO()
    topic = random.choice(["保修期多久", "维修要多久", "过保了还能修吗", "怎么开发票"])
    q = f"{p}{topic}？订单{no}"
    chosen = (f"您好！订单{no}的{p}享受官方质保，质保期内非人为损坏可免费维修或换新，"
              f"一般周期 5-7 个工作日，全程为您同步进度；如已过保我也可提供付费方案供您选择。")
    rejected = random.choice([
        "过保了就不管了。",                              # 无共情/不准确
        f"{p}终身免费保修，随便修。",                     # 乱承诺
        "不清楚，您问别人吧。",                           # 踢皮球
    ])
    return q, chosen, rejected


GENERATORS = {
    "物流查询": logistics, "退换货": return_refund, "商品咨询": consult,
    "投诉建议": complaint, "售后问题": aftersale,
}


def load_test_questions():
    if not os.path.exists(TEST_PATH):
        print(f"⚠ 未找到 {TEST_PATH}，跳过与测试集去重。")
        return set()
    qs = set()
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.add(json.loads(line).get("input", ""))
    return qs


def main():
    banned = load_test_questions()
    seen, items = set(), []
    for intent, gen in GENERATORS.items():
        got, guard = 0, 0
        while got < N_PER_INTENT and guard < N_PER_INTENT * 60:
            guard += 1
            q, chosen, rejected = gen()
            if q in seen or q in banned:        # 内部去重 + 不撞测试集
                continue
            seen.add(q)
            items.append({"question": q, "chosen": chosen,
                          "rejected": rejected, "intent": intent})
            got += 1
        print(f"{intent}: {got} 条")

    random.shuffle(items)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    leaked = sum(1 for it in items if it["question"] in banned)
    print(f"\n合计 {len(items)} 条 → {OUT_PATH}")
    print(f"与测试集重叠: {leaked} 条 {'✅ 零泄漏' if leaked == 0 else '❌ 需检查'}")


if __name__ == "__main__":
    main()