# 电商客服大模型微调项目

基于 Qwen2.5-1.5B-Instruct，针对电商客服场景完成 LoRA 微调与 DPO 偏好对齐。

## 技术路线

数据构建 → SFT微调(LoRA) → DPO对齐 → 推理部署 → 性能评测

## 环境

- GPU: NVIDIA RTX 4060Ti (8GB)
- 框架: Unsloth + TRL + Transformers
- 模型: Qwen2.5-1.5B-Instruct

## 数据

自构建电商客服数据集，覆盖5类意图：
- 物流查询 / 退换货 / 商品咨询 / 投诉建议 / 售后问题
- SFT 训练集 1800 条，测试集 200 条
- DPO 偏好数据 30 条（每类意图各 6 条，覆盖 chosen/rejected 对比）

## 训练配置

| 阶段 | 方法 | 参数 |
|------|------|------|
| SFT  | LoRA | rank=16, alpha=32, epoch=2 |
| DPO  | 偏好对齐 | beta=0.1, epoch=2, gradient_accumulation=2 |

LoRA 可训练参数：18.4M / 1562M = **1.18%**

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

### 推理性能（RTX 4060Ti，4bit量化）

| 指标 | 数值 |
|------|------|
| 平均延迟 | 1055 ms |
| P90 延迟 | 1284 ms |
| 最快响应 | 800 ms  |
| 生成速度 | 32.9 tokens/s |
| 平均输出长度 | 35 tokens |

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

**为什么用 LoRA 而不是全量微调？**
1.5B 模型全量微调需要约 12GB 显存，LoRA 只需 1.6GB，
仅训练 1.18% 的参数，显存减少 87%，效果损失极小。

**为什么先 SFT 再 DPO？**
SFT 让模型掌握客服领域知识（学会回答），
DPO 在此基础上优化回复质量（学会更好地回答）。
跳过 SFT 直接做 DPO，模型缺乏领域知识，对齐效果很差。

**SFT epoch 设置为 2 而不是 3 的原因**
epoch=3 时 train_loss 降至 0.038，模型将模板数据死记硬背导致过拟合。
epoch=2 时 train_loss 稳定在 0.199，模型保留了泛化能力，对未见过的问法更灵活。

**DPO 训练步数的重要性**
DPO 数据量少时，gradient_accumulation 过大会导致总步数不足（如仅 1~4 步），
模型无法完成偏好对齐，rewards/margins 会持续为负。
将 gradient_accumulation 从 8 降至 2，epoch 从 1 增至 2，
总步数从 4 增加到 32，rewards/margins 从 -3.3 升至 +3.0，accuracies 达到 100%。

**过拟合问题及改进方向**
当前数据为模板生成，规律性强，适当控制训练轮数可缓解过拟合。
改进方向：引入真实客服对话数据、增加数据多样性、扩充 DPO 偏好数据至 100 条以上。