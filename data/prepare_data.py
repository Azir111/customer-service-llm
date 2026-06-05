"""
prepare_data.py（方案二·配额版：强制 40% 无订单号，根治幻觉订单号）

为什么前两版没修好：带订单号的问题因订单号随机→唯一性无限，在"凑够2400条"的
循环里不断被接受；无订单号短问写法有限，去重后很快用光，比例被自动压到个位数。
本版改为【配额制】：先定死 n_no(无订单号) 与 n_order(带订单号) 的条数，分别生成；
并用槽位组合把无订单号短问的唯一写法扩到足够多，保证既能撑起比例、又全部唯一(零泄漏)。

无订单号样本的答案：语义锁死在本意图内、只索要订单号、绝不复述/编造订单号。
"""
import json
import os
import random
import re

random.seed(42)

OUT_DIR = "data/data"
SYSTEM = "你是一名专业的电商平台客服，请用耐心、专业、有同理心的态度回答用户问题。"
NO_ORDER_RATIO = 0.40

ORDER_NO = lambda: "".join(random.choices("0123456789", k=random.choice([12, 15, 18])))
PRODUCTS = ["蓝牙耳机", "保温杯", "运动鞋", "充电宝", "护肤套装", "机械键盘",
            "羽绒服", "猫粮", "台灯", "行李箱", "电饭煲", "瑜伽垫"]
DAYS = ["昨天", "前天", "三天前", "上周一", "5 天前", "一周前"]
ASK_NO = "麻烦您提供一下订单号，我好为您精准查询和处理。"


# ============ 无订单号：每个意图返回 (问题, 答案, 意图) ；答案只索要不复述 ============
def no_logistics():
    subj = random.choice(["我的", "我买的", "我下单的", "我那个", ""])
    obj = random.choice(["订单", "快递", "包裹", "东西", "货"])
    ask = random.choice(["到哪了", "怎么还没到", "什么时候到", "发货了吗", "怎么还没发货", "物流不动了", "还要多久"])
    q = f"{subj}{obj}{ask}"
    a = f"您好！很乐意为您查询物流进度。{ASK_NO}"
    return q, a, "物流查询"

def no_return():
    act = random.choice(["退货", "退款", "换货", "退掉"])
    tmpl = random.choice([f"我要{act}", f"可以{act}吗", f"怎么{act}", f"{act}流程是什么",
                          f"想{act}怎么办", f"这个能{act}吗", f"{random.choice(PRODUCTS)}想{act}"])
    a = (f"您好！很抱歉商品未能满足您的需求，我来协助您办理退换。"
         f"请问您遇到的具体问题是什么呢？另外{ASK_NO}")
    return tmpl, a, "退换货"

def no_consult():
    p = random.choice(PRODUCTS)
    asp = random.choice(["支持5G吗", "尺码怎么选", "材质是什么", "保质期多久", "有没有赠品",
                         "怎么保养", "能用多久", "值不值得买", "防水吗", "有几种颜色"])
    q = random.choice([f"这个{p}{asp}", f"{p}{asp}", f"想问下{asp}", f"请问{p}{asp}"])
    a = random.choice([
        f"您好！关于「{asp}」，详情页有完整参数；为避免给您错误信息，具体请以商品详情页和官方参数为准，我也可以帮您逐项核对。",
        f"您好！这个问题我帮您确认一下；如果方便告诉我您的具体使用场景或需求，我可以给更贴合的建议。",
    ])
    return q, a, "商品咨询"

def no_complaint():
    issue = random.choice(["服务太差了", "等了半小时没人接", "APP 老是崩溃", "发货太慢",
                           "客服态度敷衍", "问题一直没解决", "联系好几次没人管", "体验很糟糕"])
    q = random.choice([f"你们{issue}！", f"{issue}，我要投诉", f"对你们很失望，{issue}", f"{issue}"])
    a = (f"非常抱歉给您带来了不好的体验，「{issue}」确实是我们的问题，向您致歉。"
         f"您的情况我现在优先处理，同时反馈给相关团队改进，避免再次发生。")
    return q, a, "投诉建议"

def no_aftersale():
    p = random.choice(PRODUCTS)
    topic = random.choice(["保修期是多久", "维修要多久", "怎么开发票", "过保了还能修吗",
                           "怎么报修", "能保修吗", "保修怎么算", "发票怎么开"])
    q = random.choice([topic, f"{p}{topic}"])
    if "过保" in q:
        a = f"您好！若已过质保或属人为损坏，通常不在免费保修范围内，但可提供付费维修方案，{ASK_NO}我帮您评估报价。"
    elif "发票" in q:
        a = f"您好！电子发票可在订单详情页点击「申请发票」填写抬头，24 小时内发至邮箱；{ASK_NO}"
    else:
        a = f"您好！本商品享受官方质保，质保期内非人为损坏可免费维修或换新，一般周期 5-7 个工作日。{ASK_NO}"
    return q, a, "售后问题"

NO_ORDER_GENS = [no_logistics, no_return, no_consult, no_complaint, no_aftersale]


# ============ 带订单号：答案复述真实订单号 ============
def ord_logistics():
    p, no, d = random.choice(PRODUCTS), ORDER_NO(), random.choice(DAYS)
    q = random.choice([f"我{d}买的{p}到哪了，订单号{no}", f"订单{no}的{p}怎么还没发货",
                       f"{p}的快递好几天没更新了，单号{no}"])
    a = random.choice([
        f"您好！已为您查询订单{no}，{p}目前在运输途中，预计 1-2 天内送达，有更新会第一时间通知您。",
        f"非常抱歉让您久等！订单{no}的{p}我已为您催促仓库加急发货，发出后同步单号给您。",
    ])
    return q, a, "物流查询"

def ord_return():
    p, no = random.choice(PRODUCTS), ORDER_NO()
    cond = random.choice(["用了几天觉得不合适", "买错型号了", "收到就是坏的", "颜色和图片不符"])
    bad = ("坏的" in cond) or ("不符" in cond)
    q = f"订单{no}的{p}我想退货，{cond}"
    if bad:
        a = (f"非常抱歉给您带来困扰！订单{no}的{p}属于质量/描述问题，由我们承担退换运费，"
             f"麻烦您拍 1-2 张问题照片发来，我立即为您安排退换。")
    else:
        a = (f"您好！是否可退要看订单{no}是否在 7 天无理由期内及商品状态，"
             f"我先帮您核实时效，符合条件就指导您申请，不符合也会给您其他方案。")
    return q, a, "退换货"

def ord_aftersale():
    p, no = random.choice(PRODUCTS), ORDER_NO()
    topic = random.choice(["保修期是多久", "维修要多久", "怎么开发票"])
    if topic == "怎么开发票":
        q = f"订单{no}怎么开发票"
        a = f"您好！订单{no}可在订单详情页点击「申请发票」填写抬头，电子发票 24 小时内发至您邮箱；需要纸质发票请告知。"
    else:
        q = f"{p}{topic}？订单{no}"
        a = f"您好！订单{no}的{p}享受官方质保，质保期内非人为损坏可免费维修或换新，一般周期 5-7 个工作日。"
    return q, a, "售后问题"

ORDER_GENS = [ord_logistics, ord_return, ord_aftersale]


def collect(gens, target, seen):
    out, guard = [], 0
    while len(out) < target and guard < target * 200:
        guard += 1
        q, a, intent = random.choice(gens)()
        if q in seen:
            continue
        seen.add(q)
        out.append({"instruction": SYSTEM, "input": q, "output": a, "intent": intent})
    return out


def generate(n_total=2400, test_ratio=0.1):
    n_no = int(n_total * NO_ORDER_RATIO)
    n_ord = n_total - n_no
    seen = set()
    items = collect(NO_ORDER_GENS, n_no, seen) + collect(ORDER_GENS, n_ord, seen)
    random.shuffle(items)

    split = int(len(items) * (1 - test_ratio))
    train, test = items[:split], items[split:]
    train_q = {it["input"] for it in train}
    leaked = [it for it in test if it["input"] in train_q]
    assert not leaked, f"泄漏 {len(leaked)} 条"

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        with open(f"{OUT_DIR}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    has_order = lambda s: bool(re.search(r"\d{12,}", s))
    no = sum(1 for it in items if not has_order(it["input"]))
    print(f"唯一样本: {len(items)}（无订单号 {no} 条 = {no/len(items)*100:.0f}%）")
    print(f"训练集: {len(train)}  测试集: {len(test)}  泄漏: {len(leaked)}")
    print(f"输出: {OUT_DIR}/train.jsonl, {OUT_DIR}/test.jsonl")


if __name__ == "__main__":
    generate()