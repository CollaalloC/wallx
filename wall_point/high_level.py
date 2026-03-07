import cv2
import numpy as np
import os
import glob
import base64
import requests
import re
from sympy import sympify, symbols, integrate

# ==========================================
# 1. 核心图像算法 (包含阴影修正)
# ==========================================
def find_green_roi(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)

def calculate_target_list(image, result_str, spacing=40):
    roi = find_green_roi(image)
    if roi is None: return []
    x, y, w, h = roi
    cx, cy = x + w // 2, y + h // 2
    
    # 阴影修正：向右下各偏移 2 像素
    offset_x, offset_y = 2, 2
    
    n_chars = len(result_str)
    start_x = (cx + offset_x) - (n_chars - 1) * spacing // 2
    target_y = cy + offset_y
    return [(char, (int(start_x + i * spacing), int(target_y))) for i, char in enumerate(result_str)]

# ==========================================
# 2. 通用数学计算模块 (支持定积分)
# ==========================================
def universal_math_solver(expr_str):
    try:
        x = symbols('x')
        raw_res = sympify(expr_str).doit()
        res_float = float(raw_res.evalf())
        if res_float.is_integer():
            return str(int(res_float))
        else:
            return f"{res_float:.2f}"
    except Exception as e:
        print(f"   ⚠️ 数学解析异常: {expr_str}, 错误: {e}")
        return None

# ==========================================
# 3. 硅基流动 VLM 交互
# ==========================================
def call_silicon_vlm_universal(image_path, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "你是一个数学助手。请提取图片中平板上的数学算式。"
        "1. 如果是普通四则运算，输出纯算式，如: 12+5。"
        "2. 如果是定积分，请输出 Python SymPy 格式，如: integrate(x**2, (x, 0, 1))。"
        "只需输出算式代码，不要任何解释，不要空格。"
    )

    payload = {
        # Qwen/Qwen2-VL-72B-Instruct
        "model": "Qwen/Qwen3-VL-8B-Instruct", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return content.replace("`", "").strip()
        return None
    except:
        return None

# ==========================================
# 4. 单图推理主函数
# ==========================================
def process_inference(img_path, api_key):
    fname = os.path.basename(img_path)
    
    # 1. VLM 识别
    expr = call_silicon_vlm_universal(img_path, api_key)
    if not expr: return None, []

    # 2. 通用解析计算
    result_str = universal_math_solver(expr)
    if not result_str: return expr, []

    # 3. 坐标生成
    image = cv2.imread(img_path)
    planning_list = calculate_target_list(image, result_str)

    print(f"   [识别]: {expr}  ->  [计算]: {result_str}")
    
    return result_str, planning_list

# ==========================================
# 5. Main 依次调用与可视化保存
# ==========================================
if __name__ == "__main__":
    MY_KEY = "sk-kkjtucvrvosnwlrbhhoiehxdpphvhakyytqgzcvlgqocaytl"
    INPUT_DIR = "input_hl"
    OUTPUT_DIR = "output_inference"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    image_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.[jp][pn][g]*")))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # 调用推理函数
        ans_str, coords_list = process_inference(img_path, MY_KEY)
        
        # 读取原图用于可视化标注
        vis_img = cv2.imread(img_path)
        
        if coords_list:
            for char, pos in coords_list:
                # 绘制显眼的红点 (半径6, 实心)
                cv2.circle(vis_img, pos, 6, (0, 0, 255), -1)
                # 绘制白色外圈增强对比度
                cv2.circle(vis_img, pos, 7, (255, 255, 255), 1)
                
                # 在红点上方标上显眼的黄色数字
                # 坐标微调使其不遮挡红点
                text_pos = (pos[0] - 10, pos[1] - 15)
                cv2.putText(vis_img, str(char), text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 保存可视化结果
            save_path = os.path.join(OUTPUT_DIR, f"res_{filename}")
            cv2.imwrite(save_path, vis_img)
            
            print(f"🖼️ 图片: {filename}")
            print(f"✅ 答案: {ans_str} | 结果已保存至: {save_path}")
            print(f"📍 规划坐标: {coords_list}")
        else:
            print(f"🖼️ 图片: {filename} -> ❌ 处理失败")
            
        print("-" * 35)