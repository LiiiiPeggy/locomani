# SmolVLM2-256M 静态测试评估报告

- 模型: `/home/gzz/Codes/smolvlm2/models/SmolVLM2-256M-Video-Instruct`
- 设备: `cuda`
- 样本数: 22
- 总准确率: **6/22 = 27.3%**
- 模型加载: 1.06s

## 分任务准确率

| 任务 | 正确/总数 | 准确率 |
|------|-----------|--------|
| meter_reading | 1/4 | 25.0% |
| indicator_state | 1/4 | 25.0% |
| floor_water | 1/4 | 25.0% |
| window_state | 2/5 | 40.0% |
| wall_crack | 1/5 | 20.0% |

## 逐条结果

- ❌ `01_meter/meter_daylight_digits_035259.jpg` expected=`035259` pred=`03529` (0.986s)
  - raw: 03529.
- ✅ `01_meter/meter_night_digits_20509.jpg` expected=`20509` pred=`20509` (0.617s)
- ❌ `01_meter/meter_tilted_digits_135104.jpg` expected=`135104` pred=`135` (0.549s)
  - raw: 135 10.4 kwh
- ❌ `01_meter/meter_blurred_digits_53723.jpg` expected=`53723` pred=`53729` (0.503s)
  - raw: 53729
- ✅ `02_indicator/indicator_green_on.jpg` expected=`green_on` pred=`green_on` (0.69s)
- ❌ `02_indicator/indicator_red_on.jpg` expected=`red_on` pred=`green_on` (0.468s)
  - raw: green_on
- ❌ `02_indicator/indicator_off_black.jpg` expected=`off` pred=`the` (1.176s)
  - raw: The image shows a white rectangular object that is placed on the ground, possibly a sign or a piece of paper. The object
- ❌ `02_indicator/indicator_blue_color_ambiguous.jpg` expected=`uncertain` pred=`the` (0.866s)
  - raw: The image shows a car's dashboard with a blue object on the dashboard. The object is not clearly visible, but it is like
- ❌ `03_floor_water/floor_water_yes_01.jpg` expected=`yes` pred=`no` (0.387s)
  - raw: no
- ❌ `03_floor_water/floor_water_yes_02.jpg` expected=`yes` pred=`no` (0.512s)
  - raw: no
- ❌ `03_floor_water/floor_water_yes_03.jpg` expected=`yes` pred=`no` (0.396s)
  - raw: no
- ✅ `03_floor_water/floor_water_no_dry.jpg` expected=`no` pred=`no` (0.391s)
- ❌ `04_window/window_open_01.jpg` expected=`open` pred=`closed` (0.401s)
  - raw: closed
- ✅ `04_window/window_closed_01.jpg` expected=`closed` pred=`closed` (0.391s)
- ✅ `04_window/window_closed_02.jpg` expected=`closed` pred=`closed` (0.384s)
- ❌ `04_window/window_closed_building_03.jpg` expected=`closed` pred=`open` (0.428s)
  - raw: The door is open.
- ❌ `04_window/window_closed_building_04.jpg` expected=`closed` pred=`open` (0.564s)
  - raw: The door is open.
- ❌ `05_wall_crack/wall_crack_yes_01.jpg` expected=`yes` pred=`no` (0.443s)
  - raw: no
- ❌ `05_wall_crack/wall_crack_yes_02.jpg` expected=`yes` pred=`no` (0.446s)
  - raw: no
- ❌ `05_wall_crack/wall_crack_yes_03.jpg` expected=`yes` pred=`no` (0.446s)
  - raw: No
- ❌ `05_wall_crack/wall_crack_yes_04.jpg` expected=`yes` pred=`no` (0.438s)
  - raw: No
- ✅ `05_wall_crack/wall_crack_no_normal.jpg` expected=`no` pred=`no` (0.404s)

## 简要结论

本测试为 22 张 smoke test，非正式基准。结果摘要：

1. **部署层面：通过**  
   模型加载约 1s，单张推理约 0.4–1.2s，图片输入与生成流程可跑通。

2. **提示词约束：部分有效**  
   - 绿灯、部分关窗、干地面负例能按约定标签输出。  
   - 困难/模糊样本常忽略「只返回标签」约束，改为自由描述（如熄灭灯、蓝灯 ambiguous）。

3. **电表读数（P1）：弱，但非完全不可用**  
   - 4 张仅 1 张全对；另有近似错误（`035259→03529`、`53723→53729`）和截断（`135104→135`）。  
   - 说明 256M 对七段数码精细 OCR 能力不足；应按 README 建议先裁剪显示屏 ROI，或改用专用 OCR。

4. **闭集状态分类：整体偏弱，且存在系统性偏差**  
   - **积水 / 裂缝**：正例几乎全判 `no`，负例判对 → 更像「默认否定」偏差，漏检严重。  
   - **指示灯**：绿灯可对；红灯误判绿灯；熄灭/模糊样本不守格式。  
   - **窗户**：简单关窗尚可；开窗与楼宇多窗场景易混淆（甚至答成 door）。

5. **对毕设「状态识别」的含义**  
   - 开箱即用的 256M **不足以**作为工业状态识别的可靠主判决器。  
   - 可行方向：任务专用微调、ROI 裁剪、与检测/规则模型级联；把 VLM 定位为语义辅助而非唯一判据。

## 改进建议（按优先级）

1. 针对 5 类任务收集更多样本做 **LoRA/QLoRA 微调**。  
2. 电表：先检测/裁剪显示区域再问 VLM，或接轻量 OCR。  
3. 积水/裂缝：用专用分割/检测模型，VLM 仅做二次语义确认。  
4. 强化输出约束（few-shot 示例、非法输出重试）。  
5. 扩大测试集后再报告准确率，避免 22 张样本过拟合解读。
