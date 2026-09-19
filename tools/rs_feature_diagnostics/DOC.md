已完成通用遥感特征诊断工具，代码位于 [tools/rs_feature_diagnostics](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics)。

## 新增文件

核心入口与配置：

- [main.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/main.py)：分阶段 CLI 入口和完整分析流程。
- [config.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/config.py)：模型、数据、样本选择、分析上限和干预参数。
- [README.md](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/README.md)：使用说明、指标定义和边界。
- [adapter_example.yaml](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/adapter_example.yaml)：非标准模型的 hook override 模板。

模型与数据：

- [checkpoint.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/checkpoint.py)：兼容常见 checkpoint 包装和前缀，严格检查分类头。
- [dataset.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/dataset.py)：ImageFolder、timm 验证 transform、固定画布内容缩放、可选真实 mask。
- [model_inspector.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/model_inspector.py)：dry-run、结构记录、stage/feature shape 探测。
- [generic_timm.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/adapters/generic_timm.py)：feature_info、CNN/Transformer block、语义算子和 deploy 适配。
- [base.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/adapters/base.py)：Adapter 接口。

特征分析：

- [hook_manager.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/hook_manager.py)：特征采集和干预 hook。
- [tensor_adapter.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/tensor_adapter.py)：NCHW、NHWC、BNC/class-token 转换。
- [sample_selector.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/sample_selector.py)：完整验证集预测和样本筛选。
- [spatial_analysis.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/spatial_analysis.py)：energy、variance、entropy、token similarity、effective rank、区域对照。
- [scale_analysis.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/scale_analysis.py)：固定 canvas 多尺度响应和尺度偏好。
- [channel_analysis.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/channel_analysis.py)：通道相关性、尺度分组、channel ERF。
- [gradcam.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/gradcam.py)：预测类、真实类、指定类及多尺度 Grad-CAM。
- [erf.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/erf.py)：stage ERF、class-sensitive gradient、r50/r80/r90。
- [intervention.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/intervention.py)：空间/通道删除曲线及单样本三联图。
- [metrics.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/metrics.py)：指标定义与数值计算。

输出与验证：

- [paper_figures.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/paper_figures.py)：论文 Figure A/B/C。
- [failure_analysis.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/failure_analysis.py)：错分样本综合诊断图。
- [visualization.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/visualization.py)：300 DPI PNG/PDF 输出。
- [manifest.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/manifest.py)：运行环境、权重、数据、hook 和参数记录。
- [utils.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/utils.py)：CSV/JSON、随机种子、设备等公共逻辑。
- [test_diagnostics.py](/home/wnr/workspace/timm/repos/pytorch-image-models/tools/rs_feature_diagnostics/tests/test_diagnostics.py)：8 项基础测试。

没有修改现有模型或训练代码，工作区里原有的其他未提交改动也未触碰。

## 架构

执行关系为：

`CLI → ImageFolder/Checkpoint → Auto Inspector/Adapter → Hook Manager → 各分析模块 → 原始数值/论文图/Manifest`

本地实际适配的是 timm 1.0.28 的：

- `timm.create_model`
- `timm.data.resolve_model_data_config`
- `timm.data.create_transform`
- `model.feature_info`
- `model.get_classifier()`
- 本仓库 `state_dict/state_dict_ema` checkpoint 格式
- `timm.utils.model.reparameterize_model`

默认不做 deploy fusion；只有显式传入 `--deploy` 才融合并重新探测 hook。

## 第一步：Preflight

先只运行下面这一条结构检查命令：

```bash
cd /home/wnr/workspace/timm/repos/pytorch-image-models

/home/wnr/soft/miniconda3/bin/conda run -n timm \
python tools/rs_feature_diagnostics/main.py \
  --model repvit_m0_9 \
  --checkpoint output/remote_sensing/repvit_m0_9_ucm_scratch_80_20_seed0824/model_best.pth.tar \
  --data-dir /home/wnr/remote_sensing_datasets/UCM-82 \
  --val-split val \
  --device cpu \
  --analysis structure \
  --output-dir ./visualization_results
```

确认输出的 stem、stages、hook points 和 feature shapes 合理后，再正式运行。

## 正式运行

先用 `nvidia-smi` 选择空闲 GPU，然后执行：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/wnr/soft/miniconda3/bin/conda run -n timm \
python tools/rs_feature_diagnostics/main.py \
  --model repvit_m0_9 \
  --checkpoint output/remote_sensing/repvit_m0_9_ucm_scratch_80_20_seed0824/model_best.pth.tar \
  --data-dir /home/wnr/remote_sensing_datasets/UCM-82 \
  --val-split val \
  --device cuda:0 \
  --analysis all \
  --sample-mode misclassified \
  --max-samples 16 \
  --similarity-samples 16 \
  --max-tokens-for-similarity 512 \
  --intervention-samples 128 \
  --erf-channels-per-stage 16 \
  --batch-size 32 \
  --workers 4 \
  --output-dir ./visualization_results
```

分析 MobileNetV2 时只需替换：

```text
--model mobilenetv2_100
--checkpoint output/remote_sensing/mobilenetv2_100_ucm_scratch_80_20_seed0824/model_best.pth.tar
```

如果有真实二值 mask，可增加：

```text
--segmentation-mask-dir /path/to/masks
```

目录按 `<mask-dir>/<class>/<image-stem>.png` 组织。

## 验证结果

- 8 项单元测试全部通过，包括 checkpoint、layout、rank/similarity、空间/通道干预、Grad-CAM、ERF 和 mask override。
- `mobilenetv2_100`、`vit_tiny_patch16_224` 完成无网络下载的结构 dry-run。
- 两个 UCM checkpoint 均严格加载，`missing_keys=[]`、`unexpected_keys=[]`。
- MobileNetV2 和 RepViT 都使用真实 UCM 验证集完成 GPU 全链路 smoke test；每次完整预测了 420 张验证图，昂贵分析使用受限样本。
- 最终 MobileNetV2 测试产物位于 [final-mask-api-smoke](/tmp/rs-feature-diagnostics-gpu/UCM-82/mobilenetv2_100/final-mask-api-smoke)。
- smoke test 仅验证程序链路，不能作为论文统计结论；正式实验应扩大 `max-samples` 和 `intervention-samples`。

## 已知限制

- BNC token 只在 `N=H×W` 或 `H×W+1` 时自动恢复；非方形或多特殊 token 模型需要 adapter。
- 非标准 stage、复杂 residual 或自定义 attention 可能无法可靠语义识别，应先审查 preflight 并使用 YAML override。
- deploy/fused 模型的 module tree 会变化，不能沿用融合前的 hook 名称。
- 通道相关矩阵是 `O(C²)`；全通道 ERF 和大规模干预计算成本很高。
- 真实 mask 目前采用二值、ImageFolder 相对路径约定，不处理多类别 segmentation label。