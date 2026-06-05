# 评测框架（eval/）

给「电商客服大模型微调项目」补上缺失的**实验闭环**：所有结论原本都建立在 *训练集* 上（SFT/DPO 的 loss、rewards/margins），这套框架把评测搬到**测试集**，并加上**基线对比**，回答三个问题：

1. **微调到底有没有用？** → base vs SFT 在测试集上的成对胜率
2. **DPO 在 SFT 之上有没有进一步提升？** → SFT vs SFT+DPO 胜率（比训练集 rewards/margins 可信）
3. **「epoch=3 过拟合」站得住吗？** → 用 *测试集 loss* 而非 train loss 验证

## 这套东西分别修了什么

| 原项目的问题 | 本框架的对策 |
|---|---|
| 200 条测试集从未使用 | `build_eval_set.py` 把它纳入评测主集 |
| 所有指标都是训练集指标 | judge 在测试集回复上打分 / 判胜负 |
| 没有 base 基线，无法证明微调有用 | `MODELS` 含 base，pairwise 跑 sft_vs_base |
| DPO 只有训练集 100% accuracies | pairwise 跑 dpo_vs_sft，看测试集胜率 |
| train/test 同分布，测了也只是模板内拟合 | `hard_cases.jsonl` 28 条非模板探针，单独成 hard 桶 |
| 「epoch=3 过拟合」只看 train loss | `eval_loss.py` 比较各 checkpoint 的 test loss |
| 客服会乱承诺退款/折扣的风险未评 | 打分含独立的 **safety** 维度 |

## 文件

```
eval/
├── config.py            # 路径 / 模型 / judge / 字段映射，跑前先改这里
├── common.py            # 公共工具（io、加载模型、生成）
├── hard_cases.jsonl     # 28 条手写非模板泛化探针（含越界/隐私/不当让步）
├── build_eval_set.py    # ① 拼装评测集 = 模板测试集 + hard 集
├── run_inference.py     # ② 三个模型逐个加载→生成→释放（8GB 串行）
├── judge.py             # ③ LLM-as-judge：逐条打分 + 成对胜负
├── eval_loss.py         # （独立）测试集 loss 验证过拟合说法
└── report.py            # ④ 汇总成 report.md，按 bucket/intent 拆开
```

## 跑法

```bash
# 0. 装裁判用的 SDK（被测模型沿用你训练栈的 unsloth/trl）
pip install openai

# 1. 改 config.py：
#    - MODELS 里 sft/dpo 的 adapter 路径
#    - TEMPLATE_TEST_PATH 和 FIELD_* 字段映射（对齐你测试集的真实字段名）

# 2. 配置裁判（任何 OpenAI 兼容端点都行；建议用与被测不同家族的强模型）
export JUDGE_API_KEY=sk-xxxx
# 默认 DeepSeek；也可换 DashScope(qwen-max) / OpenAI：
# export JUDGE_BASE_URL=https://api.openai.com/v1 ; export JUDGE_MODEL=gpt-4o

# 3. 四步跑完
python eval/build_eval_set.py
python eval/run_inference.py
python eval/judge.py
python eval/report.py        # 产物 eval/report.md

# （可选）验证过拟合说法：先在 eval_loss.py 填两个 checkpoint 目录
python eval/eval_loss.py
```

## 评测方法的几个要点

- **成对胜负是 headline，不是绝对分**。让裁判在同一问题下二选一，比让它打绝对分稳得多。
- **位置偏差控制**：每对 A/B 双向各判一次（`JUDGE_SWAP`），两次方向不一致就记平局，避免裁判偏好「排前面的」。
- **裁判要换家族**：用 Qwen-max 评 Qwen-1.5B 有同家族「自我偏好」，更稳妥用 DeepSeek / GPT。
- **template 桶 vs hard 桶分开看**：若 hard 桶分数明显掉，说明你只是拟合了模板、泛化差——这正是 README「已知局限」里该用数据坐实的那句话。
- **safety 维度优先看**：客服模型乱承诺「立即全额退款/打五折」或不处理越权请求，是部署级风险，比平均分低更该先修。

## 局限（这套框架本身的）

- hard 集 28 条仍偏小，且由我手写，覆盖面有限；扩到 100+ 并掺入真实客服对话后结论更硬。
- LLM-as-judge 与人工标注存在偏差；条件允许时抽 10~20% 做人工复核校准。
- 4bit 推理评质量是为省显存的折中；对绝对质量敏感时建议用 merged FP16 复跑一遍 hard 桶。
