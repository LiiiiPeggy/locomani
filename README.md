# SmolVLM2 状态识别模块

基于 [HuggingFaceTB/SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) 的部署与微调方案，支持图像与视频多模态推理。

## 基础模型

| 项目 | 说明 |
|------|------|
| 模型 | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` |
| 参数量 | 256M |
| 能力 | 图像 / 视频 / 文本理解与生成 |
| 显存 | 视频推理约 1.4GB，适合 RTX 4070 8GB |
| 许可证 | Apache 2.0 |

首次使用会自动从 Hugging Face 下载权重到 `~/.cache/huggingface/`。

## 环境要求

- Python 3.10
- NVIDIA GPU + 驱动
- Conda

## 1. 获取代码

```bash
git clone https://github.com/huggingface/smollm.git
cd smollm/vision/smolvlm2
```

若已将完整仓库放在本目录下：

```bash
cd ~/Codes/smolvlm2/smollm/vision/smolvlm2
```

## 2. Conda 环境配置

```bash
conda create -n smolvlm python=3.10 -y
conda activate smolvlm
```

## 3. 安装依赖

```bash
cd ~/Codes/smolvlm2/smollm/vision/smolvlm2
cp ../../../LICENSE .
pip install -e ".[train]"
```

若尚未安装 PyTorch（已安装 cu124 可跳过）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. 部署（推理）

项目根目录提供 `inference.py`，默认加载 `SmolVLM2-256M-Video-Instruct`。

### 图像推理

```bash
conda activate smolvlm
cd ~/Codes/smolvlm2

# 正常对话模式（默认）
python inference.py \
  --image /path/to/image.jpg \
  --mode chat \
  --question "请描述画面中设备的状态。"

# 约定标签模式
python inference.py \
  --image /path/to/image.jpg \
  --mode label \
  --labels yes,no,uncertain \
  --question "地面是否有积水？"

# 任务预设（自动使用 label 模式与对应提示词）
python inference.py \
  --model_path ~/Codes/smolvlm2/models/SmolVLM2-256M-Video-Instruct \
  --image data/smolvlm2_static_test/02_indicator/indicator_green_on.jpg \
  --task indicator
```

可用 `--task`：`meter` / `indicator` / `floor_water` / `window` / `wall_crack`。

### 视频推理

```bash
python inference.py \
  --video /path/to/video.mp4 \
  --question "请判断视频中设备当前处于什么状态。"
```

### 本地文件问答（无图片）

读取工厂物体位置与状态 JSON，拼进 prompt 后用纯文本回答。示例文件：`data/example/factory_objects.json`。

```bash
python inference.py \
  --model_path ~/Codes/smolvlm2/models/SmolVLM2-256M-Video-Instruct \
  --local_file data/example/factory_objects.json \
  --question "有哪些物体？泵A的状态是什么？"
```

常用提问示例：

- `有哪些物体？请列出位置与状态。`
- `泵A在哪里？状态如何？`
- `二厂有哪些设备？分别是什么状态？`

JSON 格式要求（数组，每项至少含 `name` / `location` / `status`，可选 `factory`）：

```json
[
  {"name": "泵A", "factory": "一厂", "location": "一车间东侧", "status": "运行中"}
]
```

### 使用微调后的权重

```bash
python inference.py \
  --model_path ~/Codes/smolvlm2/smollm/vision/smolvlm2/checkpoints/state_run \
  --image /path/to/image.jpg \
  --question "设备是什么状态？"
```

### Python 代码调用

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16
).to("cuda")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "test.jpg"},
        {"type": "text", "text": "设备是什么状态？"},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt",
).to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0], skip_special_tokens=True))
```

## 5. 微调

### 5.1 准备数据

在 `data/` 下按 LLaVA 格式组织数据：

```
data/my_dataset/
├── annotations.json    # 标注文件
├── images/             # 图像
└── videos/             # 视频（可选）
```

标注示例见 `data/example/annotations.json`：

```json
{
  "type": "image",
  "image": "images/sample_001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n设备是什么状态？"},
    {"from": "gpt", "value": "运行中"}
  ]
}
```

新建数据混合配置 `scripts/mixtures/my_state.yaml`：

```yaml
datasets:
  - name: state-recognition
    modality: image
    path: .
    json_path: /home/gzz/Codes/smolvlm2/data/my_dataset/annotations.json
    sampling_strategy: all
```

### 5.2 单 GPU 微调（QLoRA，推荐 8GB 显存）

```bash
conda activate smolvlm
cd ~/Codes/smolvlm2/smollm/vision/smolvlm2
export PYTHONPATH="$PWD:$PYTHONPATH"

python smolvlm/train/train_mem.py \
  --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
  --data_mixture scripts/mixtures/my_state.yaml \
  --data_folder /home/gzz/Codes/smolvlm2/data/my_dataset \
  --output_dir checkpoints/state_run \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --language_model_lr 2e-4 \
  --connector_lr 1e-4 \
  --num_train_epochs 3 \
  --peft_enable True \
  --bits 4 \
  --gradient_checkpointing True \
  --bf16 True \
  --logging_steps 1 \
  --save_steps 100 \
  --report_to wandb \
  --run_name state_v1
```

### 5.3 全量微调（显存充足时）

```bash
python smolvlm/train/train_mem.py \
  --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
  --data_mixture scripts/mixtures/my_state.yaml \
  --data_folder /home/gzz/Codes/smolvlm2/data/my_dataset \
  --output_dir checkpoints/state_run_full \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5 \
  --language_model_lr 2e-5 \
  --connector_lr 1e-4 \
  --num_train_epochs 3 \
  --gradient_checkpointing True \
  --bf16 True \
  --report_to wandb \
  --run_name state_full_v1
```

### 5.4 Notebook 微调（备选）

也可使用官方 Notebook，基于 `transformers` Trainer + QLoRA：

- 视频：`smollm/vision/finetuning/SmolVLM2_Video_FT.ipynb`
- 图像：`smollm/vision/finetuning/Smol_VLM_FT.ipynb`

将 Notebook 中的 `model_id` 改为 `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` 即可。

## 6. 目录结构

```
smolvlm2/
├── inference.py                          # 部署推理脚本（图像/视频/本地文件问答）
├── data/
│   └── example/
│       ├── annotations.json              # 微调标注格式示例
│       └── factory_objects.json          # 本地文件问答示例
├── README.md
└── smollm/vision/smolvlm2/
    ├── scripts/mixtures/                 # 数据混合配置
    ├── smolvlm/train/train_mem.py          # 微调入口
    └── checkpoints/                      # 训练输出（运行后生成）
```

## 参考链接

- 模型权重：https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
- 官方博客：https://huggingface.co/blog/smolvlm2
- 训练代码：https://github.com/huggingface/smollm/tree/main/vision/smolvlm2
