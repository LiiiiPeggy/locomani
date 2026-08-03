#!/usr/bin/env python3
"""SmolVLM2-256M 静态测试包评估（按 data/smolvlm2_static_test/README.md）"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "smolvlm2_static_test"
DEFAULT_MODEL = ROOT / "models" / "SmolVLM2-256M-Video-Instruct"

PROMPTS = {
    "P1": (
        "读取图片中电表显示屏的数字。只返回数字，不要解释；"
        "如果无法可靠辨认，返回 unreadable。"
    ),
    "P2": (
        "判断画面中主要指示灯的颜色和亮灭状态。"
        "只返回：red_on、green_on、blue_on、off、uncertain 中的一个。"
        "如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。"
    ),
    "P3": (
        "判断地面是否存在积水或明显湿润区域。"
        "只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。"
    ),
    "P4": (
        "判断画面中主要窗户的开闭状态。"
        "只返回：open、closed、ajar、uncertain 中的一个。"
    ),
    "P5": (
        "判断墙面或混凝土表面是否存在真实裂缝。"
        "只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。"
    ),
}

ALLOWED = {
    "P2": {"red_on", "green_on", "blue_on", "off", "uncertain"},
    "P3": {"yes", "no", "uncertain"},
    "P4": {"open", "closed", "ajar", "uncertain"},
    "P5": {"yes", "no", "uncertain"},
}


def normalize_output(raw: str, prompt_id: str) -> str:
    """将模型输出转成可比较的短标签。"""
    text = raw.strip()
    # 去掉常见对话前缀
    for sep in ("Assistant:", "assistant:", "Answer:", "答案:", "回答:"):
        if sep in text:
            text = text.split(sep)[-1]
    text = text.strip().lower()
    text = re.sub(r"[`\"'。．，,！!？?\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if prompt_id == "P1":
        if "unreadable" in text:
            return "unreadable"
        digits = re.findall(r"\d+", text)
        if not digits:
            return text
        # 取最长数字串（电表读数）
        return max(digits, key=len)

    candidates = ALLOWED.get(prompt_id, set())
    # 优先完整匹配允许标签（按下划线长度优先）
    for label in sorted(candidates, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])", text):
            return label
    # 宽松：整段刚好是标签
    if text in candidates:
        return text
    return text.split(" ")[0] if text else ""


def is_correct(pred: str, expected: str, prompt_id: str) -> bool:
    if prompt_id == "P1":
        if pred == "unreadable":
            return False  # 本测试包期望具体数字；unreadable 记为未命中但合法
        # 仅比较连续数字，忽略前导零差异时可再扩展；按 README 精确比对
        return pred == expected
    return pred == expected


def extract_assistant_reply(decoded: str) -> str:
    """尽量只保留助手回复部分。"""
    for marker in ("Assistant:", "assistant:"):
        if marker in decoded:
            return decoded.split(marker)[-1].strip()
    return decoded.strip()


def run_one(processor, model, image_path: Path, question: str, device: str, max_new_tokens: int):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decoded = processor.decode(output_ids[0], skip_special_tokens=True)
    return extract_assistant_reply(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "data" / "smolvlm2_static_test" / "results"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = TEST_DIR / "manifest.csv"
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))

    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Samples: {len(rows)}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=dtype
    ).to(args.device)
    model.eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"Load time: {load_s:.2f}s")

    results = []
    by_task = {}

    for i, row in enumerate(rows, 1):
        rel = row["file"]
        task = row["task"]
        expected = row["expected"]
        prompt_id = row["prompt_id"]
        image_path = TEST_DIR / rel
        question = PROMPTS[prompt_id]

        t1 = time.perf_counter()
        try:
            raw = run_one(
                processor, model, image_path, question, args.device, args.max_new_tokens
            )
            err = ""
        except Exception as e:
            raw = ""
            err = str(e)
        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t1

        pred = normalize_output(raw, prompt_id) if not err else ""
        ok = bool(pred) and is_correct(pred, expected, prompt_id) and not err

        item = {
            "file": rel,
            "task": task,
            "prompt_id": prompt_id,
            "expected": expected,
            "raw": raw,
            "pred": pred,
            "correct": ok,
            "infer_s": round(infer_s, 3),
            "error": err,
            "notes": row.get("notes", ""),
        }
        results.append(item)

        by_task.setdefault(task, {"total": 0, "correct": 0})
        by_task[task]["total"] += 1
        if ok:
            by_task[task]["correct"] += 1

        mark = "OK" if ok else "FAIL"
        print(f"[{i:02d}/{len(rows)}] {mark} {rel} | exp={expected} pred={pred!r} ({infer_s:.2f}s)")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    summary = {
        "model_path": args.model_path,
        "device": args.device,
        "n_samples": total,
        "n_correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "load_s": round(load_s, 2),
        "by_task": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            }
            for k, v in by_task.items()
        },
        "results": results,
    }

    json_path = out_dir / "eval_results.json"
    csv_path = out_dir / "eval_results.csv"
    md_path = out_dir / "eval_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "task", "prompt_id", "expected", "pred", "correct", "infer_s", "raw", "error"],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    lines = [
        "# SmolVLM2-256M 静态测试评估报告",
        "",
        f"- 模型: `{args.model_path}`",
        f"- 设备: `{args.device}`",
        f"- 样本数: {total}",
        f"- 总准确率: **{correct}/{total} = {summary['accuracy']:.1%}**",
        f"- 模型加载: {load_s:.2f}s",
        "",
        "## 分任务准确率",
        "",
        "| 任务 | 正确/总数 | 准确率 |",
        "|------|-----------|--------|",
    ]
    for task, s in summary["by_task"].items():
        lines.append(f"| {task} | {s['correct']}/{s['total']} | {s['accuracy']:.1%} |")

    lines += ["", "## 逐条结果", ""]
    for r in results:
        mark = "✅" if r["correct"] else "❌"
        lines.append(
            f"- {mark} `{r['file']}` expected=`{r['expected']}` pred=`{r['pred']}` "
            f"({r['infer_s']}s)"
        )
        if not r["correct"] and r["raw"]:
            raw_short = r["raw"].replace("\n", " ")[:120]
            lines.append(f"  - raw: {raw_short}")

    lines += [
        "",
        "## 简要结论",
        "",
        "本测试为 22 张 smoke test，非正式基准。分任务表现可用于判断：",
        "- 闭集分类任务（指示灯/积水/窗户/裂缝）是否可用提示词约束稳定输出；",
        "- 电表读数对 256M 是否过难（若差，应尝试 ROI 裁剪或专用 OCR）。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n==== SUMMARY ====")
    print(f"Accuracy: {correct}/{total} = {summary['accuracy']:.1%}")
    for task, s in summary["by_task"].items():
        print(f"  {task}: {s['correct']}/{s['total']} = {s['accuracy']:.1%}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
