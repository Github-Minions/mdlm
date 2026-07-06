# Diffusion & MDLM 实验复现完整指南

> 本文档对应 DDPM3.pdf 中的 10 个实验，提供从零开始的复现路径、数据获取方法、代码结构与关键实现细节。
> 
> **关联仓库**：
> - 官方 MDLM 实现：`https://github.com/kuleshov-group/mdlm`
> - 官方离散扩散 guidance：`https://github.com/kuleshov-group/discrete-diffusion-guidance`
> - 参考复现（含预训练权重）：`https://github.com/goosemaths/mdlm`

---

## 目录

1. [项目结构与依赖](#1-项目结构与依赖)
2. [数据获取与准备](#2-数据获取与准备)
3. [实验总览与快速索引](#3-实验总览与快速索引)
4. [实验 1：连续二维 Toy Diffusion](#4-实验-1连续二维-toy-diffusion)
5. [实验 2：D3PM 离散状态 Toy](#5-实验-2d3pm-离散状态-toy)
6. [实验 3：CTMC Discrete Diffusion](#6-实验-3ctmc-discrete-diffusion)
7. [实验 4：MDLM 基础训练复现](#7-实验-4mdlm-基础训练复现)
8. [实验 5：采样策略消融](#8-实验-5采样策略消融)
9. [实验 6：AR/SEDD/MDLM 基线对比](#9-实验-6arseddmdlm-基线对比)
10. [实验 7：下游 Mask 修复任务](#10-实验-7下游-mask-修复任务)
11. [实验 8：Token 类型分组分析](#11-实验-8token-类型分组分析)
12. [实验 9：Top-K 交互式修复](#12-实验-9top-k-交互式修复)
13. [实验 10：D3PM vs CTMC 对比](#13-实验-10d3pm-vs-ctmc-对比)
14. [统一运行脚本](#14-统一运行脚本)
15. [常见问题排查](#15-常见问题排查)

---

## 1. 项目结构与依赖

### 1.1 代码仓库结构

```
mdlm/
|-- main.py                    # 训练/评估/采样入口
|-- diffusion.py               # 扩散过程核心（前向/反向/SUBS参数化）
|-- dataloader.py              # 数据加载（支持多种数据集）
|-- noise_schedule.py          # 噪声调度器（loglinear 等）
|-- utils.py                   # 工具函数
|-- models/                    # 骨干网络（DiT, AR, DiMamba）
|-- configs/                   # Hydra 配置文件
|   |-- data/                  # 数据集配置
|   |-- model/                 # 模型配置
|   |-- noise/                 # 噪声调度配置
|   |-- config.yaml            # 主配置（覆盖入口）
|-- scripts/                   # 训练/评估脚本
|   |-- train_owt_mdlm.sh      # OpenWebText 训练
|   |-- eval_owt_T_mdlm.sh     # 评估
|   |-- ...
|
|-- toy_experiments/           # [新增] 独立 toy 实验
|   |-- continuous_toy_diffusion.py    # 实验 1
|   |-- d3pm_toy.py                    # 实验 2
|   |-- ctmc_toy.py                    # 实验 3
|   |-- d3pm_ctmc_comparison.py        # 实验 10
|   |-- downstream_tasks.py            # 实验 7, 8
|   |-- experiment9_topk_repair.py     # 实验 9
|   |-- sampler_strategies.py          # 实验 5 辅助
|   |-- data_utils.py                  # 数据工具
|
|-- configs/data/tinystories.yaml       # [新增] TinyStories 配置
|-- configs/model/small_tinystories.yaml # [新增] 小模型配置
|-- scripts/train_tinystories_mdlm.sh   # [新增] 训练脚本
|-- scripts/eval_sampling_steps_ablation.sh # [新增] 采样消融
|-- scripts/train_baseline_comparison.sh    # [新增] 基线对比
|-- run_all_experiments.sh              # [新增] 统一运行脚本
```

### 1.2 环境安装

```bash
# 克隆复现分支
git clone -b reproduction-tinystories https://github.com/Github-Minions/mdlm.git
cd mdlm

# 创建 conda 环境（沿用官方配置）
conda env create -f requirements.yaml
conda activate mdlm

# 安装额外依赖（toy 实验 + 下游任务）
pip install scikit-learn scipy matplotlib editdistance

# 创建必要目录
mkdir -p outputs checkpoints cache
```

---

## 2. 数据获取与准备

### 2.1 TinyStories 数据集

**数据集信息**：
| 属性 | 值 |
|------|-----|
| 名称 | `roneneldan/TinyStories` |
| 训练样本 | 2,119,719 |
| 验证样本 | 21,990 |
| 下游评测 | 100 条（从验证集随机抽取，seed=1） |

**获取方式**（自动下载）：
```python
from datasets import load_dataset

# 首次运行时自动从 HuggingFace 下载
dataset = load_dataset("roneneldan/TinyStories", cache_dir="./cache")
```

**验证集固定抽取**（确保实验可复现）：
```python
import torch

# 固定 100 条验证样本
torch.manual_seed(1)
val_data = dataset['validation']
indices = torch.randperm(len(val_data))[:100].tolist()
val_texts = [val_data[i]['text'] for i in indices]
```

### 2.2 预训练权重（可选，跳过训练）

如果不想从头训练，可以直接下载 goosemaths 提供的 checkpoint：

```python
from huggingface_hub import hf_hub_download

repo_id = "goosemaths/tinystories_masked_diffusion_model_ckpt"

# MDLM step=100 checkpoint
mdlm_path = hf_hub_download(
    repo_id=repo_id,
    filename="mdlm_step100.ckpt",
    local_dir="./checkpoints"
)

# SEDD checkpoint
sedd_path = hf_hub_download(
    repo_id=repo_id,
    filename="sedd.ckpt",
    local_dir="./checkpoints"
)
```

### 2.3 词表与 Tokenizer

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# GPT-2 原生词表大小: 50257
# 需手动添加 [PAD] 和 [MASK] 特殊 token
```

---

## 3. 实验总览与快速索引

| 实验 | 内容 | 代码位置 | 依赖 |
|------|------|----------|------|
| 1 | 连续二维 toy diffusion | `toy_experiments/continuous_toy_diffusion.py` | 独立运行 |
| 2 | D3PM 离散 toy (3 种 corruption) | `toy_experiments/d3pm_toy.py` | 独立运行 |
| 3 | CTMC + tau-leaping | `toy_experiments/ctmc_toy.py` | 独立运行 |
| 4 | MDLM 基础训练 | `scripts/train_tinystories_mdlm.sh` | 官方代码 + 新增配置 |
| 5 | 采样步数/策略消融 | `scripts/eval_sampling_steps_ablation.sh` | 实验 4 checkpoint |
| 6 | AR/SEDD/MDLM 对比 | `scripts/train_baseline_comparison.sh` | 官方代码 |
| 7 | 三类 mask 修复任务 | `toy_experiments/downstream_tasks.py` | 实验 4/6 checkpoint |
| 8 | Token 类型分组准确率 | `toy_experiments/downstream_tasks.py` | 实验 7 结果分析 |
| 9 | Top-K 交互式修复 | `toy_experiments/experiment9_topk_repair.py` | 实验 4 checkpoint |
| 10 | D3PM vs CTMC 对比 | `toy_experiments/d3pm_ctmc_comparison.py` | 独立运行 |

---

## 4. 实验 1：连续二维 Toy Diffusion

### 4.1 实验目的

验证连续扩散的基本机制：前向加噪破坏二维分布结构，反向采样恢复原始分布。使用三个经典 toy 数据集。

### 4.2 数据集

| 数据集 | 特点 | 训练步数 |
|--------|------|----------|
| two_moons | 双月形，非凸 | 20,000 |
| swiss_roll | 瑞士卷流形 | 20,000 |
| gaussian_mixture | 8 组分圆环排列 | 20,000 |

### 4.3 核心公式

**前向过程（DDPM）**：
```
q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

**反向采样**：
```
x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_t) + sqrt(beta_t) * z
```

### 4.4 运行

```bash
cd toy_experiments
python continuous_toy_diffusion.py
```

### 4.5 预期输出

| 数据集 | MMD-RBF | Sliced W. | Mean Err |
|--------|---------|-----------|----------|
| two_moons | ~0.00075 | ~3.0 | ~0.024 |
| swiss_roll | ~0.00039 | ~3.7 | ~0.031 |
| gaussian_mixture | ~0.00014 | ~2.6 | ~0.030 |

---

## 5. 实验 2：D3PM 离散状态 Toy

### 5.1 实验目的

在离散 token 空间验证 D3PM 的三类 corruption 设计（uniform/absorbing/structured），比较其对恢复难度的影响。

### 5.2 实验设置

| 属性 | 值 |
|------|-----|
| 词表 | {A, B, C, D, [MASK]} |
| 序列长度 | 16 |
| 扩散步数 | 100 |
| Corruption 类型 | uniform / absorbing / structured |

### 5.3 三种 Q 矩阵设计

**Uniform**：所有转移等概率
```python
Q = torch.ones(vocab_size, vocab_size) / vocab_size
```

**Absorbing**：全部转移到 [MASK]
```python
Q[:, mask_id] = 1.0  # 每行都转移到 mask
```

**Structured**：按 token 相似度设计
```python
# A<->B, C<->D 有较高转移概率
# 保留 70% 自环概率
# 10% 概率转移到 mask
```

### 5.4 运行

```bash
cd toy_experiments
python d3pm_toy.py
```

### 5.5 预期输出

| Corruption | Val CE | Token Acc | Top-3 Acc |
|------------|--------|-----------|-----------|
| uniform | 0.659 | 70.52% | 94.03% |
| absorbing | 0.411 | 80.22% | 96.83% |
| structured | **0.328** | **85.62%** | **99.01%** |

> Structured corruption 利用 token 间结构信息，显著降低反向恢复难度。

---

## 6. 实验 3：CTMC Discrete Diffusion

### 6.1 实验目的

将离散扩散从固定时间步推广到连续时间 Markov Chain，验证 tau-leaping 的误差-效率权衡。

### 6.2 核心设计

**Forward Process**：
- Absorbing-mask CTMC，时间区间 [0, 1]
- Mask rate: lambda = 3.0
- 闭式 mask ratio: `r(t) = 1 - exp(-lambda * t)`

**Rate Matrix**：
```
R(x, y) = lambda  if x != mask, y = mask
          0       otherwise
```

**Tau-Leaping 反向采样**：
```python
for step in range(int(horizon / tau)):
    t = horizon - step * tau
    logits = model(x_t)
    # 以概率 lambda*tau 更新每个 masked 位置
```

### 6.3 Tau-Leaping 误差-效率权衡

| tau | TV Distance | NFE | Runtime |
|-----|-------------|-----|---------|
| 0.100 | 0.1923 | 10 | 0.020s |
| 0.050 | 0.0894 | 20 | 0.045s |
| 0.020 | 0.0345 | 50 | 0.116s |
| 0.010 | 0.0222 | 100 | 0.227s |
| 0.005 | 0.0111 | 200 | 0.412s |

> tau 越小，误差越低，但 NFE 和运行时间线性增长。

### 6.4 运行

```bash
cd toy_experiments
python ctmc_toy.py
```

---

## 7. 实验 4：MDLM 基础训练复现

### 7.1 实验设置

| 属性 | 值 |
|------|-----|
| 数据集 | TinyStories |
| 模型 | DiT-small (768-dim, 12 layers, 12 heads) |
| 序列长度 | 256 |
| 训练步数 | 5,000 |
| Batch size | global=128, per-GPU=16 |
| 参数化 | SUBS (substitution-based) |
| 种子 | 1 |

### 7.2 新增配置文件

**`configs/data/tinystories.yaml`**：
```yaml
# @package _global_
data:
  train: roneneldan/TinyStories
  valid: roneneldan/TinyStories
  tokenizer_name_or_path: gpt2
  cache_dir: ${cwd:}/cache
  wrap: True
```

**`configs/model/small_tinystories.yaml`**：
```yaml
name: small_tinystories
type: dit
hidden_size: 768
cond_dim: 128
length: 256
n_blocks: 12
n_heads: 12
scale_by_sigma: True
dropout: 0.1
tie_word_embeddings: False
```

### 7.3 训练命令

```bash
bash scripts/train_tinystories_mdlm.sh
```

等价命令：
```bash
python main.py \
  loader.global_batch_size=128 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small_tinystories \
  data=tinystories \
  parameterization=subs \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  eval.compute_generative_perplexity=True \
  sampling.steps=100
```

### 7.4 训练稳定性观察

- **训练 loss**：整体下降，后期进入缓慢收敛区间
- **验证 PPL**：早期存在尖峰，随后快速回落
- **Noise schedule 影响**：linear schedule 中后期波动更大，baseline 更稳定

---

## 8. 实验 5：采样策略消融

### 8.1 实验设计

固定已训练的 MDLM checkpoint，只改变推理阶段策略。

### 8.2 采样步数消融

```bash
bash scripts/eval_sampling_steps_ablation.sh
```

| Steps | Gen PPL | Runtime |
|-------|---------|---------|
| 10 | 87.02 | 10.89s |
| 20 | 51.74 | 16.13s |
| 50 | 41.77 | 14.61s |
| **100** | **30.98** | **19.41s** |
| 200 | 31.15 | 29.90s |
| 500 | 28.99 | 62.24s |
| 1000 | 28.45 | 115.68s |

> 100 步是质量-效率的较优折中。

### 8.3 Token Selection Strategy

在 `toy_experiments/sampler_strategies.py` 中实现：

| 策略 | PPL | 特点 |
|------|-----|------|
| greedy | 42.43 | 最快，但可能重复退化 |
| categorical temp=1.3 | 653.44 | 过高 temperature 破坏质量 |
| **top-k=5** | **11.81** | **最优** |
| top-k=10 | 15.20 | 次优 |

### 8.4 Unmask Strategy

| 策略 | PPL | 特点 |
|------|-----|------|
| native | 39.67 | 默认行为 |
| confidence-first | 7.90 | 较稳健 |
| left-to-right | 2.19 | PPL 极低但可能退化 |
| block | 2.19 | PPL 极低但可能退化 |

---

## 9. 实验 6：AR/SEDD/MDLM 基线对比

### 9.1 训练命令

```bash
bash scripts/train_baseline_comparison.sh
```

### 9.2 三种模型配置

| 模型 | parameterization | backbone | time_conditioning |
|------|-----------------|----------|-------------------|
| AR | `ar` | `ar` | False |
| SEDD | `sedd` | `dit` | True |
| MDLM | `subs` | `dit` | False |

### 9.3 训练结果对比

- **训练 Loss**：AR < SEDD < MDLM（AR 的 left-to-right 目标更易优化）
- **验证 PPL**：AR 最低，但 AR 不适用于 mask 修复任务
- **关键结论**：训练 loss 低不代表下游修复能力强

---

## 10. 实验 7：下游 Mask 修复任务

### 10.1 三类修复任务

| 任务 | 配置 | 测试的破坏强度 |
|------|------|----------------|
| Random Mask | 随机遮蔽 | 5%, 15%, 50% |
| OCR-like Mask | 局部 OCR 错误 | 5%, 15%, 50% |
| Span Mask | 连续缺失 | 长度 5, 10, 15 |

### 10.2 运行

```python
cd toy_experiments
python downstream_tasks.py --checkpoint ../checkpoints/mdlm_step100.ckpt
```

### 10.3 预期结果（MDLM step=100）

**Random Mask Accuracy**：
| 模型 | 5% | 15% | 50% |
|------|-----|-----|-----|
| AR | 63.90% | 57.83% | 30.96% |
| SEDD | 81.77% | 79.20% | 63.92% |
| **MDLM** | **84.42%** | **81.20%** | **65.66%** |

**Span Mask Accuracy**（length=10）：
| 模型 | Accuracy |
|------|----------|
| AR | 27.00% |
| SEDD | 32.30% |
| **MDLM** | **35.80%** |

---

## 11. 实验 8：Token 类型分组分析

在实验 7 基础上，按 token 类型分组统计准确率。

### 11.1 分类标准

| 类型 | 说明 |
|------|------|
| English | 常见英文词 |
| Punctuation | 标点符号 |
| Rare | 低频词 |

### 11.2 关键发现

- **Punctuation** 准确率通常高于 English（局部句法结构可预测）
- **Rare** token 样本数少，统计不显著
- Span Mask 场景下所有类型准确率均显著下降

---

## 12. 实验 9：Top-K 交互式修复

### 12.1 实验设计

将自动修复扩展为候选式修复：
- 高置信度位置（confidence >= 0.8）：直接自动填充
- 低置信度位置：展示 Top-K 候选供人工选择

### 12.2 运行

```python
cd toy_experiments
python experiment9_topk_repair.py --checkpoint ../checkpoints/mdlm_step100.ckpt
```

### 12.3 预期结果

| 指标 | 值 |
|------|-----|
| Top-1 Accuracy | 80.51% |
| **Top-3 Accuracy** | **92.01%** |
| Top-5 Accuracy | 94.55% |

> Top-1 到 Top-3 提升 11.50 个百分点，说明低置信位置使用候选修复具有实际价值。

---

## 13. 实验 10：D3PM vs CTMC 对比

### 13.1 实验设置

控制变量：词表、模板、Transformer 架构完全相同
变量：时间建模方式

| 方法 | 时间建模 | 反向步数 |
|------|----------|----------|
| D3PM | 80 个离散时间步 | 80 |
| CTMC | 连续时间 + tau-leaping | 160 (tau=0.00625) |

### 13.2 运行

```bash
cd toy_experiments
python d3pm_ctmc_comparison.py
```

### 13.3 关键结论

- **D3PM**：token recovery accuracy 更高（固定步数更容易训练到高逐 token 恢复率）
- **CTMC**：near-template rate 更稳定（连续时间的灵活性），但 NFE 约 2 倍

---

## 14. 统一运行脚本

```bash
bash run_all_experiments.sh
```

该脚本按顺序执行：
1. 环境检查
2. Toy 实验（1, 2, 3, 10）
3. 模型训练（4, 6）
4. 采样消融（5）
5. 下游评估（7, 8, 9）

---

## 15. 常见问题排查

| 问题 | 解决方案 |
|------|----------|
| TinyStories 下载失败 | 检查 HuggingFace 连接，或使用 `HF_ENDPOINT=https://hf-mirror.com` |
| CUDA OOM | 减小 `loader.batch_size` 或 `model.length` |
| checkpoint 加载失败 | 确保 `parameterization` 和 `backbone` 与训练时一致 |
| toy 实验 sklearn 报错 | `pip install scikit-learn scipy` |
| 生成 PPL 计算失败 | 确保 GPT-2 模型已自动下载 |
| MDLM 验证 PPL 尖峰 | 正常现象，训练早期出现，后期回落 |

---

## 附录：核心公式速查

### Mask Accuracy
```
Acc_mask = (1/|M|) * sum_{i in M} 1{x_hat_i = x_i}
```

### EDR (Edit Distance Reduction)
```
EDR = 1 - edit_distance(repaired, original) / edit_distance(corrupted, original)
```

### CTMC Mask Ratio (闭式)
```
r(t) = 1 - exp(-lambda * t)
```

### SUBS Loss (MDLM 核心)
```
L = -log p_theta(x_0 | x_t) * (dsigma / expm1(sigma))
```

---

*本文档与代码仓库 `Github-Minions/mdlm` 的 `reproduction-tinystories` 分支同步维护。*
