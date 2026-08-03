# SmolVLM2 静态状态识别测试包

本目录包含 22 张经过人工核对的静态图片，用于
`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` 部署后的首轮 smoke test。

这不是正式测试集：样本量较小，类别也不完全均衡。它的用途是验证模型加载、
图片输入、提示词约束、输出解析和基本状态识别是否能够跑通。

## 目录

- `01_meter`：4 张七段数码电表图，包括日光、夜间、倾斜和模糊条件。
- `02_indicator`：红、绿、蓝和关闭状态的 LED 面板。
- `03_floor_water`：3 张积水正例和 1 张带强反光的干地面负例。
- `04_window`：1 张打开和 4 张关闭状态的窗户。
- `05_wall_crack`：4 张裂缝正例和 1 张正常墙面负例。
- `manifest.csv`：每张图片的预期答案、来源和许可证。

## 建议提示词

### P1：仪表读数

```text
读取图片中电表显示屏的数字。只返回数字，不要解释；如果无法可靠辨认，返回 unreadable。
```

`manifest.csv` 中的仪表答案来自原数据集的数字级 COCO 标注。原标注不包含小数点，
因此这里测试的是连续数字识别，不评价单位或小数点。

### P2：指示灯

```text
判断画面中主要指示灯的颜色和亮灭状态。
只返回：red_on、green_on、blue_on、off、uncertain 中的一个。
如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。
```

### P3：地面积水

```text
判断地面是否存在积水或明显湿润区域。
只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。
```

### P4：窗户状态

```text
判断画面中主要窗户的开闭状态。
只返回：open、closed、ajar、uncertain 中的一个。
```

### P5：墙面裂缝

```text
判断墙面或混凝土表面是否存在真实裂缝。
只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
```

## 测试建议

1. 先使用上面的任务专用提示词测试，避免让 256M 模型自由描述整幅图片。
2. 推理时关闭采样，例如 `do_sample=False`，便于重复比较。
3. 将输出转成小写并去除标点后，再和 `manifest.csv` 中的预期答案比较。
4. 对 `unreadable`、`uncertain` 视为合法答案；模型在困难图片上拒绝猜测通常比错误自信更有价值。
5. 如果模型能跑通但抄表表现差，先裁剪显示屏区域再测试，以区分视觉分辨率问题和部署问题。

## 数据来源

- 电表：[Mendeley 七段数码电表数据集](https://data.mendeley.com/datasets/fnn44p4mj8/1)，CC0 1.0。
- 指示灯：[Roboflow RGB LED panel](https://universe.roboflow.com/inspiration-robotx-workspace/rgb-led-panel)，CC BY 4.0。
- 积水：[Roboflow Water Puddles](https://universe.roboflow.com/projet-industrie/water-puddles-du0za)，CC BY 4.0。
- 窗户：[Roboflow Window detection](https://universe.roboflow.com/smart-buildings/window-detection-tzxgz)，CC BY 4.0；其余窗户图片来自 Wikimedia Commons，具体页面见 `manifest.csv`。
- 裂缝：[Roboflow Crack](https://universe.roboflow.com/university-bswxt/crack-bphdr)，Public Domain；正常墙面来自 Wikimedia Commons CC0 图片。

