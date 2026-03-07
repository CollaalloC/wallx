# 在dataset文件夹外直接读取video和parquet文件，处理后输出到pointed_video文件夹
import cv2
import numpy as np
import os
import glob
import pandas as pd
import shutil
import re
# ==========================================
# 1. 核心图像算法 (适配 360p)
# ==========================================
def find_green_roi(image):
    """提取绿色答题区边界框 (360p 优化版)"""
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

def calculate_target_dot(ref_frame, last_frame):
    """计算第5帧与最后一帧的差异中心"""
    roi = find_green_roi(ref_frame)
    if roi is None: return None
    x, y, w, h = roi
    
    # 差分分析
    gray_ref = cv2.cvtColor(ref_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_last, gray_ref)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 清除边缘噪点
    margin = 5
    cv2.rectangle(thresh, (0, 0), (w-1, h-1), 0, thickness=margin)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    target_contour = max(contours, key=cv2.contourArea)
    
    # 1. 计算带有阴影偏离的原始几何中心
    M = cv2.moments(target_contour)
    if M["m00"] != 0:
        raw_cx = int(M["m10"] / M["m00"])
        raw_cy = int(M["m01"] / M["m00"])
    else:
        # 如果矩计算失败，使用外接矩形中心作为备选
        bx, by, bw, bh = cv2.boundingRect(target_contour)
        raw_cx, raw_cy = bx + bw // 2, by + bh // 2

    # ==========================================================
    # 2. 【阴影偏移修正】针对 360p 数字块左上阴影
    # 由于阴影使中心偏左上，我们需要向 右下 补偿。
    # ==========================================================
    offset_x = 6  # 向右修正像素数 (正数向右)
    offset_y = 16  # 向下修正像素数 (正数向下)
    
    # 应用修正
    cx = raw_cx + offset_x
    cy = raw_cy + offset_y
    
    # 对最终坐标进行边界检查，防止修正后超出 ROI 区域 (可选，更鲁棒)
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    print(f"   DEBUG: 原始中心({raw_cx}, {raw_cy}), 修正后({cx}, {cy})")

    # 返回相对于全画面的坐标
    return (x + cx, y + cy)

# ==========================================
# 2. 数据集处理逻辑
# ==========================================
def process_lerobot_v3_dataset(root_dir):
    data_dir = os.path.join(root_dir, "data")
    video_root = os.path.join(root_dir, "videos", "observation.images.fixed")
    output_video_root = os.path.join(root_dir, "pointed_video")

    # 递归查找所有 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    print(f"🚀 启动！检测到 {len(parquet_files)} 个数据分片。")

    for pq_path in parquet_files:
        # 1. 获取当前 parquet 所在的 chunk 文件夹名 (例如: chunk-000)
        chunk_name = os.path.basename(os.path.dirname(pq_path))
        
        # 2. 获取文件名 (例如: file-000)
        file_base = os.path.basename(pq_path).replace(".parquet", "")
        
        # 3. 拼接完整的视频路径
        # 路径结构: data_50/videos/observation.images.fixed/chunk-000/file-000.mp4
        mp4_path = os.path.join(video_root, chunk_name, f"{file_base}.mp4")
        
        # 4. 对应的输出路径也要保持 chunk 结构
        out_dir = os.path.join(output_video_root, chunk_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f"{file_base}.mp4")

        if not os.path.exists(mp4_path):
            print(f"⚠️ 找不到视频: {mp4_path}")
            continue

        print(f"\n🎬 成功匹配: {chunk_name}/{file_base}")
        print(f"\n🎬 正在分析分片: {file_base}")
        df = pd.read_parquet(pq_path)
        
        # --- Pass 1: 预计算该视频内每个 Episode 的打点坐标 ---
        episode_coords = {}
        cap = cv2.VideoCapture(mp4_path)
        
        # 按 episode_index 分组查找
        grouped = df.groupby('episode_index')
        for ep_idx, group in grouped:
            # 获取该 episode 在 parquet 中的所有行索引
            # 注意：如果 parquet 索引不是从0开始，建议使用 group.index.tolist()
            frame_indices = group.index.tolist()
            
            if len(frame_indices) < 10: 
                print(f"   ⚠️ Episode {ep_idx} 帧数不足，跳过计算。")
                continue

            # 定位第 5 帧 (索引位置 4) 和 最后一帧
            idx_5 = frame_indices[4]
            idx_last = frame_indices[-1]

            # 抓取第 5 帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_5)
            ret1, frame_5 = cap.read()
            # 抓取最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_last)
            ret2, frame_last = cap.read()

            if ret1 and ret2:
                coord = calculate_target_dot(frame_5, frame_last)
                if coord:
                    episode_coords[ep_idx] = coord
                    print(f"   📍 Episode {ep_idx} (帧 {idx_5}-{idx_last}) 坐标锁定: {coord}")

        # --- Pass 2: 重新压制并实时打点 ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = os.path.join(output_video_root, f"{file_base}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print(f"   ⏳ 正在渲染新视频...")
        # 逐帧遍历 Parquet 记录进行渲染
        for i in range(len(df)):
            ret, frame = cap.read()
            if not ret: break
            
            # 获取当前帧所属的 episode_index
            curr_ep = df['episode_index'].iloc[i]
            target = episode_coords.get(curr_ep)
            
            if target:
                # 绘制 360p 优化红点：红心白边
                cv2.circle(frame, target, 3, (0, 0, 255), -1)
                cv2.circle(frame, target, 4, (255, 255, 255), 1)
            
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"   ✅ 分片 {file_base}.mp4 处理完成。")

    print(f"\n🎉 任务全部完成！处理后的视频保存在: {output_video_root}")

if __name__ == "__main__":
    # 配置 data_50 文件夹路径
    DATA_SET_ROOT = "data_300" 
    
    if os.path.exists(DATA_SET_ROOT):
        process_lerobot_v3_dataset(DATA_SET_ROOT)
    else:
        print(f"错误: 找不到目录 {DATA_SET_ROOT}")