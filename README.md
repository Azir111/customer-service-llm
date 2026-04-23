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
- 训练集 1800 条，测试集 200 条

## 训练配置

| 阶段 | 方法 | 参数 |
|------|------|------|
| SFT  | LoRA | rank=16, alpha=32, epoch=3 |
| DPO  | 偏好对齐 | beta=0.1, epoch=1 |

LoRA 可训练参数：18.4M / 1562M = **1.18%**

## 实验结果

### Loss 曲线
| 阶段 | 初始loss | 最终loss |
|------|---------|---------|
| SFT  | 2.901   | 0.038   |

### 推理性能（RTX 4060Ti，4bit量化）

| 指标 | 数值 |
|------|------|
| 平均延迟 | 1051 ms |
| P90 延迟 | 1519 ms |
| 最快响应 | 764 ms  |
| 生成速度 | 33.9 tokens/s |

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
1.5B模型全量微调需要约12GB显存，LoRA只需1.6GB，
仅训练1.18%的参数，显存减少87%，效果损失极小。

**为什么先SFT再DPO？**
SFT让模型掌握客服领域知识（学会回答），
DPO在此基础上优化回复质量（学会更好地回答）。
跳过SFT直接做DPO，模型缺乏领域知识，对齐效果很差。

**过拟合问题及改进方向**
当前数据为模板生成，规律性强导致模型过拟合。
改进方向：引入真实客服对话数据、增加数据多样性、
适当减少训练轮数。
