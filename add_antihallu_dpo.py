"""
add_antihallu_dpo.py —— 给 DPO 数据补一批"反幻觉订单号"偏好对，治贪心下仍编订单号的问题。

每条偏好对：
  question = 无订单号的短问（用户没给订单号）
  chosen   = 不编号、礼貌索要订单号、语义不串味（正确行为）
  rejected = 凭空编造一个订单号 + 常见的"串味成物流"（正是当前模型的真实失败输出）
DPO 会把梯度朝"远离 rejected(编号/串味)"推，比 SFT 调比例可靠。

合并进现有 data/data/dpo_data.json（去重），不覆盖你已有的偏好数据。
"""
import json
import os
import random

random.seed(11)

DPO_PATH = "data/data/dpo_data.json"
N_PER_INTENT = 16

ORDER_NO = lambda: "".join(random.choices("0123456789", k=random.choice([12, 15])))
PRODUCTS = ["蓝牙耳机", "保温杯", "运动鞋", "充电宝", "护肤套装", "机械键盘",
            "羽绒服", "猫粮", "台灯", "行李箱", "电饭煲", "瑜伽垫"]
ASK = "麻烦您提供一下订单号，我好为您精准查询和处理。"


def logistics():
    q = random.choice(["我的订单到哪了", "怎么还没发货啊", "我的快递什么时候到",
                       "帮我查下物流", "东西寄出来了吗", "还要多久能到", "发货了没"])
    chosen = f"您好！很乐意为您查询物流进度。{ASK}"
    rejected = f"您好！已为您查询订单{ORDER_NO()}，{random.choice(PRODUCTS)}预计 1-2 天内送达。"
    return q, chosen, rejected, "物流查询"


def return_refund():
    act = random.choice(["退货", "退款", "换货"])
    q = random.choice([f"我要{act}", f"可以{act}吗", f"怎么{act}", f"{act}流程是什么", f"想{act}怎么办"])
    chosen = (f"您好！很抱歉商品未能满足您的需求，我来协助您办理退换。"
              f"请问您遇到的具体问题是什么呢？另外{ASK}")
    rejected = random.choice([
        f"您好！订单{ORDER_NO()}的{random.choice(PRODUCTS)}怎么还没发货，三天前就下单了",  # 串味成物流
        f"您好！订单{ORDER_NO()}是否可退要看是否在 7 天无理由期内。",                       # 编订单号
    ])
    return q, chosen, rejected, "退换货"


def aftersale():
    q = random.choice(["保修期是多久", "维修要多久", "怎么开发票", "能保修吗", "怎么报修"])
    chosen = (f"您好！本商品享受官方质保，质保期内非人为损坏可免费维修或换新，一般周期 5-7 个工作日。{ASK}")
    rejected = f"您好！订单{ORDER_NO()}的{random.choice(PRODUCTS)}享受官方质保，质保期内可免费维修。"
    return q, chosen, rejected, "售后问题"


GENS = [logistics, return_refund, aftersale]


def main():
    # 读已有 DPO 数据
    existing = []
    if os.path.exists(DPO_PATH):
        existing = json.load(open(DPO_PATH, encoding="utf-8"))
    seen = {it["question"] for it in existing}

    added = []
    for gen in GENS:
        got, guard = 0, 0
        while got < N_PER_INTENT and guard < N_PER_INTENT * 80:
            guard += 1
            q, chosen, rejected, intent = gen()
            if q in seen:
                continue
            seen.add(q)
            added.append({"question": q, "chosen": chosen, "rejected": rejected, "intent": intent})
            got += 1

    merged = existing + added
    json.dump(merged, open(DPO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"原有 {len(existing)} 条 + 新增反幻觉 {len(added)} 条 = 共 {len(merged)} 条 → {DPO_PATH}")
    print("\n新增样例：")
    for it in added[:3]:
        print(f"  问: {it['question']}")
        print(f"  ✓ chosen  : {it['chosen'][:40]}...")
        print(f"  ✗ rejected: {it['rejected'][:40]}...")
        print()


if __name__ == "__main__":
    main()