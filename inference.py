#!/usr/bin/env python3
"""
SmolVLM2-256M-Video-Instruct 推理脚本（图像 / 视频 / 本地文件问答）

输出模式：
  --mode chat   正常对话，自由文本回答
  --mode label  约定标签，尽量只返回给定标签之一
"""
import argparse
import json
################################
# 计时：输出模型加载与推理耗时
import time
################################
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_LOCAL_QUESTION = "有哪些物体？请列出位置与状态。"

################################
# 输出模式：chat=正常对话；label=约定标签
# task 预设对应静态测试包中的 P1–P5 提示词
TASK_PRESETS = {
    "meter": {
        "question": (
            "读取图片中电表显示屏的数字。只返回数字，不要解释；"
            "如果无法可靠辨认，返回 unreadable。"
        ),
        "labels": None,  # 数字任务，无闭集标签
        "hint": "只返回数字或 unreadable，不要解释。",
    },
    "indicator": {
        "question": (
            "判断画面中主要指示灯的颜色和亮灭状态。"
            "如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。"
        ),
        "labels": ["red_on", "green_on", "blue_on", "off", "uncertain"],
        "hint": None,
    },
    "floor_water": {
        "question": (
            "判断地面是否存在积水或明显湿润区域。"
            "不要把普通反光直接判定为积水。"
        ),
        "labels": ["yes", "no", "uncertain"],
        "hint": None,
    },
    "window": {
        "question": "判断画面中主要窗户的开闭状态。",
        "labels": ["open", "closed", "ajar", "uncertain"],
        "hint": None,
    },
    "wall_crack": {
        "question": (
            "判断墙面或混凝土表面是否存在真实裂缝。"
            "不要把画框边缘、阴影或纹理当成裂缝。"
        ),
        "labels": ["yes", "no", "uncertain"],
        "hint": None,
    },
}
################################


def load_model(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ################################
    # transformers 新版本弃用 torch_dtype，改用 dtype
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
    ).to(device)
    ################################
    model.eval()
    return processor, model


################################
# 按输出模式改写用户问题，并配置 generate 参数
def build_question(question: str, mode: str, labels=None, hint: str = None) -> str:
    """chat：原样；label：附加只返回标签的约束。"""
    if mode == "chat":
        return question

    parts = [question.strip()]
    if labels:
        label_str = "、".join(labels)
        parts.append(f"只返回以下标签之一：{label_str}。不要解释，不要输出其它文字。")
    elif hint:
        parts.append(hint)
    else:
        parts.append("只返回简短约定标签或结论，不要解释。")
    return "\n".join(parts)


def generate_kwargs(mode: str, max_new_tokens: int) -> dict:
    """label 模式关闭采样、限制长度，提高格式稳定性。"""
    if mode == "label":
        return {
            "max_new_tokens": min(max_new_tokens, 64),
            "do_sample": False,
        }
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }


def extract_assistant_reply(decoded: str) -> str:
    for marker in ("Assistant:", "assistant:"):
        if marker in decoded:
            return decoded.split(marker)[-1].strip()
    return decoded.strip()


def run_generate(processor, model, messages, device: str, mode: str, max_new_tokens: int):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs(mode, max_new_tokens))
    return extract_assistant_reply(
        processor.decode(output_ids[0], skip_special_tokens=True)
    )
################################


################################
# 本地文件问答：读取工厂物体 JSON 并拼进 prompt
def load_local_objects(path: str):
    """读取本地物体状态 JSON，要求为 list[dict]，且含 name/location/status。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"本地状态文件不存在: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"本地状态文件必须是 JSON 数组: {path}")

    required = ("name", "location", "status")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条记录必须是对象(dict)")
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(f"第 {i} 条记录缺少字段: {', '.join(missing)}")

    return data


def format_objects_for_prompt(objects):
    """将物体列表格式化为可读文本。"""
    lines = []
    for obj in objects:
        parts = [f"名称: {obj['name']}"]
        if obj.get("factory"):
            parts.append(f"工厂: {obj['factory']}")
        parts.append(f"位置: {obj['location']}")
        parts.append(f"状态: {obj['status']}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def build_local_file_prompt(objects, question: str) -> str:
    info = format_objects_for_prompt(objects)
    return (
        "你是工厂设备状态助手。请仅根据下面提供的本地物体信息回答，"
        "不要编造未给出的信息。\n\n"
        "【本地物体信息】\n"
        f"{info}\n\n"
        "【用户问题】\n"
        f"{question}"
    )
################################


def run_text(processor, model, prompt: str, device: str, mode: str, max_new_tokens: int):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return run_generate(processor, model, messages, device, mode, max_new_tokens)


def run_image(
    processor, model, image_path: str, question: str, device: str, mode: str, max_new_tokens: int
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    return run_generate(processor, model, messages, device, mode, max_new_tokens)


def run_video(
    processor, model, video_path: str, question: str, device: str, mode: str, max_new_tokens: int
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video", "path": video_path},
            ],
        }
    ]
    return run_generate(processor, model, messages, device, mode, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="SmolVLM2 推理")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL,
        help="HF 模型 ID 或本地微调权重目录",
    )
    parser.add_argument("--image", help="图像路径")
    parser.add_argument("--video", help="视频路径")
    ################################
    # 传入 --local_file 即开启读本地工厂物体状态文件模式（纯文本，无图片）
    parser.add_argument(
        "--local_file",
        help="本地工厂物体状态 JSON 路径（开启后不输入图片/视频）",
    )
    ################################
    ################################
    # 输出模式：chat=正常对话；label=约定标签
    parser.add_argument(
        "--mode",
        choices=["chat", "label"],
        default=None,
        help="输出模式：chat=正常对话；label=只返回约定标签（指定 --task 时默认 label，否则默认 chat）",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="label 模式下的允许标签，逗号分隔，如 yes,no,uncertain",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_PRESETS.keys()),
        default=None,
        help="可选任务预设（meter/indicator/floor_water/window/wall_crack），会设置问题与标签",
    )
    ################################
    parser.add_argument(
        "--question",
        default=None,
        help="提问内容（与 --task 同时指定时，以 --question 为准）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ################################
    # 模式校验：local_file 与 image/video 互斥；三者至少其一
    modes = [bool(args.image), bool(args.video), bool(args.local_file)]
    if sum(modes) == 0:
        parser.error("请指定 --image、--video 或 --local_file 之一")
    if sum(modes) > 1:
        parser.error("--image、--video、--local_file 不能同时使用")

    labels = None
    hint = None
    if args.task:
        preset = TASK_PRESETS[args.task]
        if args.question is None:
            args.question = preset["question"]
        labels = preset["labels"]
        hint = preset.get("hint")

    # 未显式指定 --mode 时：有 task 默认 label，否则 chat
    if args.mode is None:
        args.mode = "label" if args.task else "chat"

    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    if args.question is None:
        if args.local_file:
            args.question = DEFAULT_LOCAL_QUESTION
        else:
            args.question = "请描述画面中的设备状态。"

    if args.mode == "label" and not labels and not hint and not args.task:
        hint = "只返回简短约定结论，不要解释。"

    question = build_question(args.question, args.mode, labels=labels, hint=hint)
    ################################

    ################################
    # 分别统计模型加载耗时与推理耗时
    t0 = time.perf_counter()
    processor, model = load_model(args.model_path, args.device)
    if device_is_cuda := (args.device == "cuda" and torch.cuda.is_available()):
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.local_file:
        objects = load_local_objects(args.local_file)
        prompt = build_local_file_prompt(objects, question)
        result = run_text(
            processor, model, prompt, args.device, args.mode, args.max_new_tokens
        )
    elif args.image:
        result = run_image(
            processor,
            model,
            args.image,
            question,
            args.device,
            args.mode,
            args.max_new_tokens,
        )
    else:
        result = run_video(
            processor,
            model,
            args.video,
            question,
            args.device,
            args.mode,
            args.max_new_tokens,
        )
    if device_is_cuda:
        torch.cuda.synchronize()
    infer_s = time.perf_counter() - t1
    total_s = load_s + infer_s
    ################################

    print(result)
    ################################
    print(
        f"\n[模式] {args.mode}"
        + (f" | labels={labels}" if labels else "")
        + f"\n[耗时] 加载: {load_s:.2f}s | 推理: {infer_s:.2f}s | 合计: {total_s:.2f}s"
    )
    ################################


if __name__ == "__main__":
    main()
