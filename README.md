# Visualization
计算机视觉深度学习中常用的可视化工具包
# RS Feature Diagnostics

面向遥感图像分类骨干网络的**特征可视化、受控干预与性能瓶颈诊断工具**。

本项目的目标不是简单生成 Grad-CAM 或特征热力图，而是帮助回答更具体的问题：

> 当前模型的性能瓶颈更可能来自空间信息、通道表示、尺度/上下文处理，还是目前的证据还不足以判断？

工具以同一验证集、同一样本和可追踪的模型输入为基础，联合使用分类错误分析、空间/通道诊断、尺度干预、Grad-CAM、ERF 和受控特征干预，并保存可审计的原始结果与离线报告。

---

## Features

### Classification diagnosis

* 全验证集 Top-1
* Confusion Matrix
* Per-class Precision / Recall / F1
* 高频错分类别对
* Reliability / ECE / NLL
* Risk-Coverage Curve
* Corrected / degraded / unchanged sample tracking

### Spatial diagnosis

* Stage-wise spatial energy
* Spatial variance / entropy
* Token similarity
* Effective rank
* Homogeneous-region feature similarity
* Grad-CAM
* Stage ERF
* Spatial feature deletion

空间干预支持与**相同保留数量的随机位置干预**比较，避免仅根据“删除后性能变化”直接推断空间重要性。

### Channel diagnosis

* Channel cosine / Pearson similarity
* Channel redundancy
* Channel scale preference
* Channel ERF
* Similarity-cluster intervention
* Random-channel control

通道删除实验记录实际保留通道数和有效 retained ratio，可与随机删除进行对照。

### Scale / context diagnosis

工具将常见的“缩放图像”进一步拆分为：

* **Content scale**：目标/内容在固定输入画布中的占用大小变化
* **Resolution-only**：只改变有效分辨率
* **Context-only**：减少周围上下文，但不放大保留像素
* 多位置 crop control

因此，“放大后预测变正确”不会直接被解释为模型存在尺度问题。

### Auditable preprocessing

所有模型解释结果共享统一的输入几何处理：

```text
Original Image
      ↓
Resize
      ↓
Evaluation Crop
      ↓
Model Input Canvas
```

Grad-CAM、ERF、空间特征图和 segmentation mask 均基于实际模型输入几何进行对齐。

每个样本保存稳定 `sample_id`、relative path 和输入 fingerprint，用于跨运行和跨模型的严格样本配对。

---

# Quick Start

## 1. Create a configuration

```bash
python -m tools.rs_feature_diagnostics init \
    --output configs/my_experiment.yaml
```

生成的 YAML 包含模型、checkpoint、数据集、预处理、分析范围与输出设置。

编辑其中的关键路径：

```yaml
schema_version: 2

experiment: aid_baseline_diagnosis

model_source:
  python_paths:
    - /path/to/pytorch-image-models

models:
  baseline:
    name: rs_dinet_baseline_l1_d4
    checkpoint: /path/to/model_best.pth.tar
    state: model

data:
  root: /path/to/AID
  train_split: train
  val_split: val

runtime:
  device: cuda:0
  batch_size: 32
  workers: 4

output:
  root: ./diagnostic_results
```

---

## 2. Check the experiment before running

```bash
python -m tools.rs_feature_diagnostics doctor \
    --config configs/my_experiment.yaml
```

先确认：

* 数据集路径
* checkpoint 路径
* 外部模型源码路径
* device
* 分析范围
* 预计分析样本数

对于自动 stage 识别不可靠的模型，应使用 adapter override 明确指定 hook 位置。

---

## 3. Run diagnostics

```bash
python -m tools.rs_feature_diagnostics run \
    --config configs/my_experiment.yaml
```

一次完整运行会区分三种样本范围：

```text
Full Population
    └── predictions/predictions.csv

Diagnostic Cohort
    └── predictions/diagnostic_samples.csv

Display Cohort
    └── predictions/display_samples.csv
```

全集结果用于整体统计；较大的 diagnostic cohort 用于诊断；较小的 display cohort 用于生成可阅读的案例图。

---

## 4. Open the offline report

运行结束后打开：

```text
<run_dir>/index.html
```

同时会生成：

```text
SUMMARY.md
SUMMARY_DATA.json
evidence.jsonl
artifacts.jsonl
case_index.csv
```

这些文件用于区分：

* 直接观察
* 受控干预结果
* 样本转移
* 当前限制
* 尚不能判断的问题

---

# Summarize an Existing Run

无需重新加载模型：

```bash
python -m tools.rs_feature_diagnostics summarize \
    --run /path/to/run
```

工具会从已经保存的结构化证据生成确定性摘要，并同时生成：

```text
prompts/SUMMARIZE_THIS_RUN.md
```

该提示词可以交给能够读取运行目录与图片的 AI 助手做进一步总结。

自动总结必须区分：

```text
Observed fact
Controlled comparison
Hypothesis
Insufficient evidence
```

不会仅根据 CAM 亮区、特征相关性或一次干预结果直接推荐新的网络模块。

---

# Compare Two Models

Baseline 与候选模型分别完成诊断后，可按稳定 `sample_id` 比较：

```bash
python -m tools.rs_feature_diagnostics compare \
    --config configs/compare.yaml
```

比较器要求两个运行具有完全一致的样本身份与真实标签，否则拒绝比较。

结果将样本划分为：

```text
stable_correct
corrected
degraded
unchanged_wrong
```

并计算：

```text
net_gain = corrected - degraded
```

这比只比较最终 Top-1 更适合分析一个新模块究竟纠正了什么、又破坏了什么。

---

# Output Structure

典型运行目录：

```text
run/
├── manifest/
├── predictions/
├── classification/
├── samples/
├── gradcam/
├── attention/
├── spatial/
├── scale/
├── erf/
├── channel/
├── interventions/
├── paper_figures/
├── tables/
├── prompts/
│
├── SUMMARY.md
├── SUMMARY_DATA.json
├── evidence.jsonl
├── artifacts.jsonl
├── case_index.csv
└── index.html
```

详细 CSV / JSON / NPY 数据用于复查和重新绘图，不建议仅凭单张图片作结论。

---

# Diagnostic Principle

本工具遵循一个简单原则：

> **先证明当前模型存在什么问题，再决定是否需要修改网络。**

例如：

```text
错分
 ↓
是否存在稳定的空间干扰？
 ↓
空间干预是否优于随机位置干预？
 ↓
正确样本是否同时被破坏？
 ↓
不同 Stage 是否表现一致？
 ↓
证据充分后，再考虑空间建模方法
```

类似地，通道相关性高并不自动说明模型需要通道注意力；缩放后分类恢复也不自动说明模型需要多尺度模块。

如果现有实验无法区分多个解释，工具应该输出：

> **Insufficient evidence**

而不是强行给出结构设计结论。

---

# Supported Models

工具主要围绕 `timm` 风格分类模型设计，同时提供：

* CNN `NCHW`
* NHWC feature
* Transformer `BNC`
* class-token layouts
* global-token Transformer attention rollout
* custom adapter override

无法可靠映射到空间网格的 token 或 window/cross-attention 不会强行生成误导性的图，而是明确记录为 skipped。

---

# Important Limitations

* Grad-CAM 和 attention visualization 主要用于提出假设，不单独构成因果证据。
* 删除/替换中间特征属于推理干预，并不等价于重新训练后的消融实验。
* Homogeneous-region analysis 默认只是 RGB texture proxy，不是真实前景/背景标签。
* 没有真实 segmentation mask 时，不应把纹理平坦区域称为 background。
* Dataset-wide interventions、channel ERF 等分析计算成本较高。
* 自动 stage discovery 无法保证适配所有自定义网络。
* 理论特征冗余不等价于真实 GPU / CPU 加速。

---

# Legacy CLI

旧版 CLI 仍保留用于兼容历史实验：

```bash
python tools/rs_feature_diagnostics/main.py --help
```

新实验建议优先使用 schema-versioned workflow：

```bash
python -m tools.rs_feature_diagnostics ...
```

---

# Project Status

当前项目处于研究工具持续完善阶段。

优先目标是：

1. 保证诊断结果与真实模型输入严格对应；
2. 保证跨模型比较使用相同样本；
3. 将相关性观察与受控干预分开；
4. 将重要结果组织成易于人工检查的组图；
5. 允许最终结论为“当前证据不足”。

如果你发现分析结果与模型实际行为不一致，欢迎提交 issue，并附上运行目录中的 `run_manifest.json` 与相关 evidence/table 文件。
