# 电商客服大模型微调项目

基于 Qwen2.5-1.5B-Instruct，针对电商客服场景完成 **QLoRA 微调**与 **DPO 偏好对齐**。

> 关联仓库：[LLM 高并发推理服务 & 性能压测平台](#) —— 同一个模型的工程化部署与性能归因。
> 本仓库关注**模型能力**（学会回答、对齐回复质量），压测仓库关注**工程能力**（高并发服务化 + 性能评测）。

## 技术路线

数据构建 → SFT微调(QLoRA) → DPO对齐 → 推理部署 → 性能评测

## 环境

- GPU: NVIDIA RTX 4060Ti (8GB)
- 框架: Unsloth + TRL + Transformers
- 模型: Qwen2.5-1.5B-Instruct

## 数据

自构建电商客服数据集，覆盖 5 类意图：
- 物流查询 / 退换货 / 商品咨询 / 投诉建议 / 售后问题
- SFT 训练集 1800 条，测试集 200 条
- DPO 偏好数据 30 条（每类意图各 6 条，覆盖 chosen/rejected 对比）

## 训练配置

| 阶段 | 方法 | 参数 |
|------|------|------|
| SFT  | QLoRA（4bit NF4 基座 + LoRA adapter） | rank=16, alpha=32, epoch=2 |
| DPO  | 偏好对齐（QLoRA） | beta=0.1, epoch=2, gradient_accumulation=2 |

LoRA 可训练参数：18.4M / 1562M = **1.18%**（基座 4bit 加载并冻结，仅训练 adapter）

## 实验结果

### Loss 曲线

| 阶段 | 初始 loss | 最终 loss | 说明 |
|------|---------|---------|------|
| SFT  | 2.901   | 0.199   | epoch=2，健康收敛，无过拟合 |
| DPO  | —       | 0.579   | margins 从 -3.3 升至 +3.0，方向正确 |

### DPO 对齐效果

| 指标 | 数值 |
|------|------|
| rewards/margins | +3.055 |
| rewards/accuracies | 100% |
| rewards/chosen | +2.058 |
| rewards/rejected | -0.997 |

> ⚠️ 说明：accuracies 100% 是在 30 条偏好数据上的**训练集指标**，数据量小、模板规律性强，不代表泛化对齐效果。真实质量评测见「改进方向」。

### 推理性能（RTX 4060Ti，4bit 量化）

| 指标 | 数值 |
|------|------|
| 平均延迟 | 1055 ms |
| P90 延迟 | 1284 ms |
| 最快响应 | 800 ms  |
| 生成速度 | 32.9 tokens/s |
| 平均输出长度 | 35 tokens |

> 本表为**本仓库 4bit 单请求**推理结果。高并发 / FP16 下的吞吐与延迟归因，见压测仓库。

## 模型精度说明（训练 vs 部署）

为避免「训练用 4bit、部署却是 FP16」的常见误解，这里把模型的精度生命周期讲清楚：

| 阶段 | 精度 | 说明 |
|------|------|------|
| 训练（QLoRA） | 基座 4bit（NF4） + adapter 高精度 | 4bit 仅为训练省显存的手段，基座冻结不更新 |
| 本仓库推理（serve.py） | 4bit | 直接以 4bit 加载，对应上方「推理性能」表（1055ms / 32.9 tok/s） |
| 合并导出 merged_model | **FP16** | merge LoRA 时基座反量化回 16bit 再焊上 adapter；实测 `model.safetensors` 为 2.9GB，与 1.5B×2字节≈3GB 吻合 |
| GGUF 导出（export_model.py） | 4bit（q4_k_m） | 内部经 FP16 中间态后量化落盘，供 llama.cpp / Ollama / LM Studio 本地部署 |
| 压测仓库部署 | FP16 | A/B 组加载 FP16 merged_model，C 组另行量化 4bit 作朴素基线 |

**要点：4bit 是训练阶段为塞进 8GB 显存的临时手段，不是模型的最终身份。**
合并后产出的是 FP16 权重（已实测 2.9GB 验证）；下游部署用 4bit 还是 FP16，按各自的显存 / 延迟 / 精度约束分别决定，不必与训练一致。

> 量化方法辨析：训练用的 NF4（bitsandbytes）与 GGUF 导出用的 q4_k_m（llama.cpp）都叫「4bit」，但量化算法、分组方式、精度损失特性不同，分属训练生态与本地推理生态，不可混为一谈。

## 快速开始

```bash
# 安装依赖
pip install unsloth trl transformers datasets

# 准备数据
python data/prepare_data.py

# SFT 微调
python train_sft.py

# DPO 对齐
python train_dpo.py

# 推理测试
python serve.py
```

## 关键技术点

**为什么用 QLoRA（而非普通 LoRA / 全量微调）?**

全量微调 1.5B 要把 FP16 基座 + 全部参数的梯度和优化器状态都放进显存，约 12GB，单张 8GB 卡放不下。
QLoRA 用两件事叠加省显存：
1. 基座以 4bit（NF4）加载并冻结，权重体积压到约 1/4；
2. 只训练占 1.18% 的 LoRA adapter（18.4M 参数），其余权重不参与梯度更新。

两者叠加把训练显存压到约 1.6GB（相比全量约 12GB 降低约 87%），从而能在 RTX 4060Ti（8GB）上完成微调，效果损失很小。

> 术语澄清：普通 LoRA 的基座是 FP16，光基座就约 3GB，压不到 1.6GB——能到 1.6GB 正是因为基座做了 4bit 量化，所以准确叫法是 **QLoRA**（Q = Quantized 基座 + LoRA adapter）。代码中体现为 `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`。

**为什么先 SFT 再 DPO?**

SFT 让模型掌握客服领域知识（学会回答），DPO 在此基础上优化回复质量（学会更好地回答）。
跳过 SFT 直接做 DPO，模型缺乏领域知识，对齐效果很差。

**SFT epoch 设置为 2 而不是 3 的原因**

epoch=3 时 train_loss 降至 0.038，模型把模板数据死记硬背，导致过拟合。
epoch=2 时 train_loss 稳定在 0.199，模型保留了泛化能力，对未见过的问法更灵活。

**DPO 训练步数的重要性**

DPO 数据量少时，gradient_accumulation 过大会导致总步数不足（如仅 1~4 步），模型无法完成偏好对齐，rewards/margins 会持续为负。
将 gradient_accumulation 从 8 降至 2、epoch 从 1 增至 2，总步数从 4 增加到 32，rewards/margins 从 -3.3 升至 +3.0，accuracies 达到 100%。

**过拟合问题及改进方向**

当前数据为模板生成、规律性强，适当控制训练轮数可缓解过拟合，但根本上受限于数据多样性。

## 已知局限与改进方向

诚实标注当前版本的边界，避免把训练集指标误读为泛化能力：

- **缺独立质量评测**：当前只有训练过程指标（loss、rewards/margins），尚无独立测试集上的回答质量评测。计划加小规模评测集 + LLM-as-judge，验证对齐后回答质量真正提升而非仅在训练集上拟合。
- **数据规模与多样性不足**：模板生成数据规律性强，DPO 仅 30 条。计划引入真实客服对话、扩充 DPO 偏好数据至 100 条以上，并报告测试集而非训练集指标。
- **DPO accuracies 100% 需谨慎解读**：该数字来自 30 条训练数据，样本过小，不能作为对齐效果的最终结论。
