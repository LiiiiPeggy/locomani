# SmolVLM2-256M 静态测试：每张图片问答明细

模型：`/home/gzz/Codes/smolvlm2/models/SmolVLM2-256M-Video-Instruct`  
数据目录：`data/smolvlm2_static_test/`  
说明：下列「问」为评估时实际输入的提示词；「答」为模型原始输出；「期望」为 `manifest.csv` 中的人工核对答案。

---

## 01 电表读数（P1）

### 1. `01_meter/meter_daylight_digits_035259.jpg`

- **问**：读取图片中电表显示屏的数字。只返回数字，不要解释；如果无法可靠辨认，返回 unreadable。
- **答**：`03529.`
- **期望**：`035259`
- **判定**：错误

### 2. `01_meter/meter_night_digits_20509.jpg`

- **问**：读取图片中电表显示屏的数字。只返回数字，不要解释；如果无法可靠辨认，返回 unreadable。
- **答**：`3624 20509 kwh`
- **期望**：`20509`
- **判定**：正确（解析后取到 `20509`）

### 3. `01_meter/meter_tilted_digits_135104.jpg`

- **问**：读取图片中电表显示屏的数字。只返回数字，不要解释；如果无法可靠辨认，返回 unreadable。
- **答**：`135 10.4 kwh`
- **期望**：`135104`
- **判定**：错误

### 4. `01_meter/meter_blurred_digits_53723.jpg`

- **问**：读取图片中电表显示屏的数字。只返回数字，不要解释；如果无法可靠辨认，返回 unreadable。
- **答**：`53729`
- **期望**：`53723`
- **判定**：错误

---

## 02 指示灯（P2）

### 5. `02_indicator/indicator_green_on.jpg`

- **问**：判断画面中主要指示灯的颜色和亮灭状态。只返回：red_on、green_on、blue_on、off、uncertain 中的一个。如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。
- **答**：`green_on`
- **期望**：`green_on`
- **判定**：正确

### 6. `02_indicator/indicator_red_on.jpg`

- **问**：判断画面中主要指示灯的颜色和亮灭状态。只返回：red_on、green_on、blue_on、off、uncertain 中的一个。如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。
- **答**：`green_on`
- **期望**：`red_on`
- **判定**：错误

### 7. `02_indicator/indicator_off_black.jpg`

- **问**：判断画面中主要指示灯的颜色和亮灭状态。只返回：red_on、green_on、blue_on、off、uncertain 中的一个。如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。
- **答**：`The image shows a white rectangular object that is placed on the ground, possibly a sign or a piece of paper. The object is not illuminated, and there is no visible light source. The background is a plain white wall, and the ground is covered with a light-colored, possibly beige or brown, ground cover`
- **期望**：`off`
- **判定**：错误（未按约定标签输出）

### 8. `02_indicator/indicator_blue_color_ambiguous.jpg`

- **问**：判断画面中主要指示灯的颜色和亮灭状态。只返回：red_on、green_on、blue_on、off、uncertain 中的一个。如果只能看出外壳颜色但无法判断是否发光，返回 uncertain。
- **答**：`The image shows a car's dashboard with a blue object on the dashboard. The object is not clearly visible, but it is likely a light source that illuminates the dashboard.`
- **期望**：`uncertain`
- **判定**：错误（未按约定标签输出）

---

## 03 地面积水（P3）

### 9. `03_floor_water/floor_water_yes_01.jpg`

- **问**：判断地面是否存在积水或明显湿润区域。只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。
- **答**：`no`
- **期望**：`yes`
- **判定**：错误

### 10. `03_floor_water/floor_water_yes_02.jpg`

- **问**：判断地面是否存在积水或明显湿润区域。只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。
- **答**：`no`
- **期望**：`yes`
- **判定**：错误

### 11. `03_floor_water/floor_water_yes_03.jpg`

- **问**：判断地面是否存在积水或明显湿润区域。只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。
- **答**：`no`
- **期望**：`yes`
- **判定**：错误

### 12. `03_floor_water/floor_water_no_dry.jpg`

- **问**：判断地面是否存在积水或明显湿润区域。只返回：yes、no、uncertain 中的一个。不要把普通反光直接判定为积水。
- **答**：`no`
- **期望**：`no`
- **判定**：正确

---

## 04 窗户状态（P4）

### 13. `04_window/window_open_01.jpg`

- **问**：判断画面中主要窗户的开闭状态。只返回：open、closed、ajar、uncertain 中的一个。
- **答**：`closed`
- **期望**：`open`
- **判定**：错误

### 14. `04_window/window_closed_01.jpg`

- **问**：判断画面中主要窗户的开闭状态。只返回：open、closed、ajar、uncertain 中的一个。
- **答**：`closed`
- **期望**：`closed`
- **判定**：正确

### 15. `04_window/window_closed_02.jpg`

- **问**：判断画面中主要窗户的开闭状态。只返回：open、closed、ajar、uncertain 中的一个。
- **答**：`closed`
- **期望**：`closed`
- **判定**：正确

### 16. `04_window/window_closed_building_03.jpg`

- **问**：判断画面中主要窗户的开闭状态。只返回：open、closed、ajar、uncertain 中的一个。
- **答**：`The door is open.`
- **期望**：`closed`
- **判定**：错误

### 17. `04_window/window_closed_building_04.jpg`

- **问**：判断画面中主要窗户的开闭状态。只返回：open、closed、ajar、uncertain 中的一个。
- **答**：`The door is open.`
- **期望**：`closed`
- **判定**：错误

---

## 05 墙面裂缝（P5）

### 18. `05_wall_crack/wall_crack_yes_01.jpg`

- **问**：判断墙面或混凝土表面是否存在真实裂缝。只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
- **答**：`no`
- **期望**：`yes`
- **判定**：错误

### 19. `05_wall_crack/wall_crack_yes_02.jpg`

- **问**：判断墙面或混凝土表面是否存在真实裂缝。只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
- **答**：`no`
- **期望**：`yes`
- **判定**：错误

### 20. `05_wall_crack/wall_crack_yes_03.jpg`

- **问**：判断墙面或混凝土表面是否存在真实裂缝。只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
- **答**：`No`
- **期望**：`yes`
- **判定**：错误

### 21. `05_wall_crack/wall_crack_yes_04.jpg`

- **问**：判断墙面或混凝土表面是否存在真实裂缝。只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
- **答**：`No`
- **期望**：`yes`
- **判定**：错误

### 22. `05_wall_crack/wall_crack_no_normal.jpg`

- **问**：判断墙面或混凝土表面是否存在真实裂缝。只返回：yes、no、uncertain 中的一个。不要把画框边缘、阴影或纹理当成裂缝。
- **答**：`No`
- **期望**：`no`
- **判定**：正确

---

## 汇总

| 正确 | 错误 | 准确率 |
|------|------|--------|
| 6 | 16 | 27.3% |

原始结果文件：`eval_results.csv` / `eval_results.json`
