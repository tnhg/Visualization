"""Offline evidence package, HTML renderer, and constrained summary prompt."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import ArtifactRecord, EvidenceRecord
from .utils import artifact_id, write_csv, write_json


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
        return value if value == value and abs(value) != float("inf") else None
    except (KeyError, TypeError, ValueError):
        return None


def _artifact_type(path: Path) -> str:
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return "figure"
    if path.suffix.lower() in {".npy", ".npz"}:
        return "raw_array"
    if path.suffix.lower() in {".csv", ".json", ".jsonl"}:
        return "table"
    return "document"


def build_artifact_index(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"artifacts.jsonl", "index.html"}:
            continue
        relative = path.relative_to(run_dir).as_posix()
        record = ArtifactRecord(
            artifact_id=artifact_id(relative), artifact_type=_artifact_type(path),
            path=relative, status="complete", scope="population" if relative.startswith("predictions/") else "diagnostic",
            raw_array=relative if path.suffix.lower() in {".npy", ".npz"} else None,
        )
        records.append(record.to_dict())
    target = run_dir / "artifacts.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    return records


def build_evidence_package(run_dir: Path, language: str = "zh") -> dict[str, Any]:
    """Derive deterministic facts from existing tables; never infer causes."""
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "manifest" / "run_manifest.json")
    status = _read_json(run_dir / "manifest" / "run_status.json")
    cls = _read_json(run_dir / "tables" / "classification_summary.json")
    attention_rows = _read_csv(run_dir / "tables" / "attention_rollout_manifest.csv")
    erf_rows = _read_csv(run_dir / "tables" / "erf_radii.csv")
    scale_rows = _read_csv(run_dir / "tables" / "scale_intervention_summary.csv")
    transitions = _read_csv(run_dir / "tables" / "scale_intervention_predictions.csv")
    evidence: list[dict[str, Any]] = []
    if cls:
        accuracy = cls.get("top1_accuracy_percent", 100 * float(cls.get("top1_accuracy", 0)))
        evidence.append(EvidenceRecord(
            "E001", "observation", f"全集基础预测 Top-1 为 {float(accuracy):.3f}%。",
            int(cls.get("samples", 0)), int(cls.get("samples", 0)), "全验证集基础预测",
            "tables/classification_summary.json", "top1_accuracy", ["classification/confusion_counts.png"],
            ["这是分类相关性统计，不说明热图或干预具有因果性。"], "observation").to_dict())
    if scale_rows:
        ranked = sorted(
            (row for row in scale_rows if row.get("condition") != "content_scale_1"),
            key=lambda row: (_float(row, "net_gain") if _float(row, "net_gain") is not None else float("-inf")),
            reverse=True,
        )
        if ranked:
            row = ranked[0]
            evidence.append(EvidenceRecord(
                "E002", "controlled_intervention",
                f"条件 {row.get('condition')} 的纠正 {row.get('corrected', '')}、退化 {row.get('degraded', '')}，净变化 {row.get('net_gain', '')}。",
                int(float(row.get("samples", 0))), int(float(row.get("originally_wrong", 0))),
                "同一批原始错误样本的内容/控制变换", "tables/scale_intervention_summary.csv",
                f"condition={row.get('condition')}", ["scale/scale_sensitivity.png"],
                ["这是受控推理干预，不是重新训练消融，也不能单独推出模块设计。"], "comparison").to_dict())
    if transitions:
        counts: dict[str, int] = {}
        for row in transitions:
            key = row.get("transition", "unknown")
            counts[key] = counts.get(key, 0) + 1
        evidence.append(EvidenceRecord(
            "E003", "transition", "已保存每个样本的稳定 transition 配对记录：" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            len(transitions), len(transitions), "before/after 逐图配对", "tables/scale_intervention_predictions.csv",
            "transition", [], ["若样本池不是全集，不能把该计数当作全验证集比例。"], "comparison").to_dict())
    limitations = []
    if not cls:
        limitations.append("尚未生成全集分类摘要。")
    if not transitions:
        limitations.append("尚未生成尺度或对照干预配对。")
    if manifest.get("checkpoint_report", {}).get("unsafe_pickle_used"):
        limitations.append("checkpoint 使用了显式信任的 legacy pickle，结果应按运行清单审阅。")
    if manifest.get("missing_semantic_ops"):
        limitations.append("部分语义操作未被 adapter 确认，相关高级分析可能 skip。")
    skipped_attention = sum(1 for row in attention_rows if str(row.get("status", "")).startswith("skipped"))
    if skipped_attention:
        limitations.append(f"attention rollout 有 {skipped_attention} 个样本因模型布局不支持而 skip。")
    unavailable_erf = sum(1 for row in erf_rows if row.get("status") == "unavailable")
    if unavailable_erf:
        limitations.append(f"ERF 有 {unavailable_erf} 个查询无有效梯度，未解释为大感受野。")
    report_status = status.get("status", "unknown")
    if limitations and report_status == "complete":
        report_status = "partial"
    summary_data = {
        "schema_version": 2, "status": report_status,
        "run_dir": run_dir.name, "manifest": manifest,
        "facts": evidence, "limitations": limitations,
        "next_checks": ["若需要跨模型结论，请先用同一 sample_id 集合运行 compare；当前摘要不自动推断原因。"],
    }
    write_json(run_dir / "SUMMARY_DATA.json", summary_data)
    (run_dir / "evidence.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in evidence), encoding="utf-8")
    build_artifact_index(run_dir)
    selected = _read_csv(run_dir / "predictions" / "selected_samples.csv")
    case_schema = (
        "sample_id", "dataset_index", "relative_path", "image_path", "true_class", "true_index",
        "pred_class", "pred_index", "confidence", "correct")
    write_csv(run_dir / "case_index.csv", [
        {key: row.get(key) for key in case_schema} for row in selected], schema=case_schema)
    facts_md = "\n".join(f"- **{row['evidence_id']}** {row['fact']}（来源：`{row['source_file']}` / `{row['row_key']}`）" for row in evidence)
    limits_md = "\n".join(f"- {item}" for item in limitations) or "- 当前没有额外限制记录。"
    summary = "# 运行摘要\n\n## 确定性事实\n\n" + (facts_md or "- 当前没有可引用事实。") + "\n\n## 当前限制\n\n" + limits_md + "\n"
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    prompt = _summary_prompt(run_dir)
    (run_dir / "prompts" / "SUMMARIZE_THIS_RUN.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts" / "SUMMARIZE_THIS_RUN.md").write_text(prompt, encoding="utf-8")
    return summary_data


def _summary_prompt(run_dir: Path) -> str:
    return f"""# 总结诊断运行（证据约束）

请先读取 `SUMMARY_DATA.json` 和 `evidence.jsonl`，再按需打开 `index.html` 中最多 3 张相关组图。运行目录：`{run_dir.name}`。

用自然中文先给结论，最多列出 3 条发现。每条严格写成“现象—对照—限制”，并引用 evidence_id 及其 source_file/row_key。区分观察、对照、假设和不能判断；没有图像访问时只总结表格并说明限制。不要把相关性、CAM 亮区、mask 稀疏或一次干预曲线写成因果或真实加速结论。最多提出一个低成本下一步检查，不自动建议重新训练或设计模块。
"""


def render_run(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    data = _read_json(run_dir / "SUMMARY_DATA.json")
    if not data:
        data = build_evidence_package(run_dir)
    facts = data.get("facts", [])
    limitations = data.get("limitations", [])
    links: list[str] = []
    for path in sorted(run_dir.rglob("*.png")):
        relative = path.relative_to(run_dir).as_posix()
        if len(links) >= 12:
            break
        links.append(f'<figure><img loading="lazy" src="{html.escape(relative)}"><figcaption>{html.escape(relative)}</figcaption></figure>')
    fact_html = "".join(f"<li><b>{html.escape(str(row.get('evidence_id')))}</b> {html.escape(str(row.get('fact')))}<br><small>{html.escape(str(row.get('source_file')))} / {html.escape(str(row.get('row_key')))}</small></li>" for row in facts)
    limit_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations) or "<li>无额外限制记录。</li>"
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>遥感诊断报告</title>
<style>body{{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#20242a;background:#fff}}h1,h2{{color:#17324d}}.status{{padding:.7rem 1rem;background:#eef5fb;border-left:4px solid #3277ad}}figure{{margin:.5rem;display:inline-block;vertical-align:top;width:30%}}figure img{{max-width:100%;height:180px;object-fit:contain;background:#f4f5f6}}figcaption{{font-size:.75rem;overflow-wrap:anywhere}}small{{color:#64748b}}</style></head><body>
<h1>遥感特征诊断离线报告</h1><div class="status">状态：{html.escape(str(data.get('status','unknown')))}；报告不依赖网络，可直接用浏览器打开。</div>
<h2>最值得先看的证据</h2><ol>{fact_html or '<li>暂无证据记录。</li>'}</ol>
<h2>当前限制</h2><ul>{limit_html}</ul>
<h2>组图（最多展示 12 张）</h2><div>{''.join(links) or '<p>暂无 PNG 组图。</p>'}</div>
<h2>机器可读文件</h2><ul><li><a href="SUMMARY.md">SUMMARY.md</a></li><li><a href="SUMMARY_DATA.json">SUMMARY_DATA.json</a></li><li><a href="evidence.jsonl">evidence.jsonl</a></li><li><a href="artifacts.jsonl">artifacts.jsonl</a></li><li><a href="prompts/SUMMARIZE_THIS_RUN.md">SUMMARIZE_THIS_RUN.md</a></li></ul></body></html>"""
    target = run_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def compare_prediction_runs(baseline_run: Path, candidate_run: Path, output_dir: Path) -> dict[str, Any]:
    """Join two completed prediction populations by stable sample_id."""
    baseline_rows = _read_csv(Path(baseline_run) / "predictions" / "predictions.csv")
    candidate_rows = _read_csv(Path(candidate_run) / "predictions" / "predictions.csv")
    if any(not row.get("sample_id") for row in (*baseline_rows, *candidate_rows)):
        raise ValueError("compare requires a non-empty stable sample_id on every prediction row")
    baseline = {row["sample_id"]: row for row in baseline_rows}
    candidate = {row["sample_id"]: row for row in candidate_rows}
    if len(baseline) != len(baseline_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("compare requires unique stable sample_id values within each run")
    if not baseline or not candidate:
        raise ValueError("compare requires both runs to contain predictions/predictions.csv")
    shared = sorted(set(baseline) & set(candidate))
    if len(shared) != len(baseline) or len(shared) != len(candidate):
        raise ValueError(f"sample populations differ: baseline={len(baseline)}, candidate={len(candidate)}, shared={len(shared)}")
    rows: list[dict[str, Any]] = []
    counts = {key: 0 for key in ("stable_correct", "corrected", "degraded", "unchanged_wrong")}
    for sample_id in shared:
        left, right = baseline[sample_id], candidate[sample_id]
        target_left, target_right = left.get("true_index"), right.get("true_index")
        left_path = left.get("relative_path") or left.get("image_path")
        right_path = right.get("relative_path") or right.get("image_path")
        if target_left != target_right or left_path != right_path:
            raise ValueError(f"sample mapping differs for {sample_id}")
        before = str(left.get("pred_index")) == target_left
        after = str(right.get("pred_index")) == target_right
        transition = "stable_correct" if before and after else "degraded" if before else "corrected" if after else "unchanged_wrong"
        counts[transition] += 1
        rows.append({
            "sample_id": sample_id, "relative_path": left_path, "image_path": left.get("image_path"), "true_index": target_left,
            "baseline_pred_index": left.get("pred_index"), "candidate_pred_index": right.get("pred_index"),
            "baseline_confidence": left.get("confidence"), "candidate_confidence": right.get("confidence"),
            "transition": transition,
        })
    output_dir = Path(output_dir).resolve()
    (output_dir / "comparison").mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison" / "transition.csv", rows)
    summary = {
        "schema_version": 2, "baseline_run": str(Path(baseline_run).resolve()),
        "candidate_run": str(Path(candidate_run).resolve()), "samples": len(shared),
        **counts, "net_gain": counts["corrected"] - counts["degraded"],
    }
    write_json(output_dir / "comparison" / "SUMMARY_DATA.json", summary)
    return summary
