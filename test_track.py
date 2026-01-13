import torch
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import (
    load_and_preprocess_images, 
    convert_coords_with_transform
)
from vggt.utils.visual_track import visualize_tracks_on_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(dtype)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
checkpoint_path = "./model.pt"
state_dict = torch.load(checkpoint_path)
model = VGGT()
model.load_state_dict(state_dict)
model = model.to(dtype=dtype)
model.eval()
model = model.to(device)

# Load all images from a specified folder
image_folder = "/home/geneta/dataset/source_dataset/euroc/V1_03_difficult/V103_change"  # 修改为你的图片文件夹路径

# Get all image files from the folder
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
image_names = []
for ext in image_extensions:
    image_names.extend(Path(image_folder).glob(f'*{ext}'))
    image_names.extend(Path(image_folder).glob(f'*{ext.upper()}'))

# Sort by filename to ensure consistent order
image_names = sorted([str(p) for p in image_names])

if len(image_names) == 0:
    raise ValueError(f"No images found in folder: {image_folder}")

print(f"Found {len(image_names)} images in folder: {image_folder}")
print(f"Image files: {[os.path.basename(name) for name in image_names]}")

# 加载图像并获取坐标转换对象（使用双向映射）
print("\nLoading and preprocessing images...")
load_start_time = time.time()
images, transforms = load_and_preprocess_images(image_names, mode="crop", return_transforms=True)
load_time = time.time() - load_start_time
print(f"Image loading time: {load_time:.3f}s")

images = images.to(device)

# Convert from (N, C, H, W) to (1, N, C, H, W) where N is sequence length
# aggregator expects shape [B, S, C, H, W] where B=batch, S=sequence
if len(images.shape) == 4:
    images = images.unsqueeze(0)  # Add batch dimension

print(f"Images shape: {images.shape}")  # Should be (1, N, 3, H, W)

# 原始图像中的像素位置（例如：原始图像尺寸为 1920x1080，你想跟踪的点是 (960, 540)）
# 修改这里的坐标为你的原始图像中的像素位置
original_query_point = torch.FloatTensor([[723.0, 224.0]])  # 原始图像坐标 [x, y]

# 使用双向映射转换为预处理后的坐标（更高效）
preprocessed_query_point = convert_coords_with_transform(
    original_query_point,
    transforms[0],  # 使用第一张图像的转换对象
    to_preprocessed=True
)

print(f"Original query point: {original_query_point[0].tolist()}")
print(f"Preprocessed query point: {preprocessed_query_point[0].tolist()}")

# 开始推理计时
print("\n" + "="*60)
print("Starting inference...")
print("="*60)
inference_start_time = time.time()

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Aggregator阶段
        agg_start = time.time()
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        agg_time = time.time() - agg_start
        print(f"Aggregator time: {agg_time:.3f}s")
        
        query_points = preprocessed_query_point.to(device)

        # 深度图预测阶段
        depth_start = time.time()
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        depth_time = time.time() - depth_start
        print(f"Depth prediction time: {depth_time:.3f}s")
        
        # 转换为numpy并移除batch维度
        depth_map_np = depth_map.squeeze(0).cpu().numpy()  # (S, H, W, 1)
        depth_conf_np = depth_conf.squeeze(0).cpu().numpy()  # (S, H, W)
        
        # 移除最后一个维度（如果是1）
        if depth_map_np.shape[-1] == 1:
            depth_map_np = depth_map_np.squeeze(-1)  # (S, H, W)
        
        # 追踪预测阶段
        track_start = time.time()
        track, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
        track_time = time.time() - track_start
        print(f"Tracking time: {track_time:.3f}s")

# 总推理时间
total_inference_time = time.time() - inference_start_time
print("="*60)
print(f"Total inference time: {total_inference_time:.3f}s")
print(f"  - Aggregator: {agg_time:.3f}s ({agg_time/total_inference_time*100:.1f}%)")
print(f"  - Depth prediction: {depth_time:.3f}s ({depth_time/total_inference_time*100:.1f}%)")
print(f"  - Tracking: {track_time:.3f}s ({track_time/total_inference_time*100:.1f}%)")
print(f"Average time per frame: {total_inference_time/len(image_names):.3f}s")
print("="*60)

print(f"\nDepth map shape: {depth_map.shape}")
print(f"Depth confidence shape: {depth_conf.shape}")
print(f"Track shape: {track.shape}")
print(f"Visibility scores shape: {vis_score.shape}")
print(f"Confidence scores shape: {conf_score.shape}")

# Visualize tracking results
# Use preprocessed images for visualization since track coordinates are in preprocessed image space
# images shape is (1, S, 3, H, W), convert to (S, 3, H, W)
images_for_viz = images.squeeze(0).cpu()  # (S, 3, H, W)

# Prepare tracks: track is (1, S, N, 2), we need (S, N, 2)
# vis_score is (1, S, N), we need (S, N)
tracks_for_viz = track.squeeze(0).cpu()  # (S, N, 2)
vis_mask = (vis_score.squeeze(0).cpu() > 0.5)  # (S, N) - threshold visibility

# 使用双向映射将追踪坐标转换回原始图像坐标（更高效）
print("\nConverting tracks back to original image coordinates...")
coord_convert_start = time.time()
tracks_original = []
for s in range(tracks_for_viz.shape[0]):
    # 获取当前帧的追踪坐标 (N, 2)
    frame_tracks = tracks_for_viz[s]  # (N, 2)
    # 使用双向映射转换回原始图像坐标
    original_tracks = convert_coords_with_transform(
        frame_tracks,
        transforms[s],  # 使用对应帧的转换对象
        to_preprocessed=False
    )
    tracks_original.append(original_tracks)

# 堆叠所有帧的原始坐标 (S, N, 2)
tracks_original = torch.stack(tracks_original)
coord_convert_time = time.time() - coord_convert_start
print(f"Coordinate conversion time: {coord_convert_time:.3f}s")
print(f"Original tracks shape: {tracks_original.shape}")
print(f"Sample original track coordinates (first point, first frame): {tracks_original[0, 0].tolist()}")

# Create output directory
output_dir = os.path.join(image_folder, "track_visualization")

print(f"\nVisualizing tracks and saving to: {output_dir}")

# Enhanced visualization with trajectory lines
S, N, _ = tracks_for_viz.shape
H, W = images_for_viz.shape[2], images_for_viz.shape[3]

# Get track colors
from vggt.utils.visual_track import get_track_colors_by_position
track_colors_rgb = get_track_colors_by_position(
    tracks_for_viz,
    vis_mask_b=vis_mask,
    image_width=W,
    image_height=H,
    cmap_name="hsv"
)

# Create visualization frames with trajectory lines
frame_images = []
os.makedirs(output_dir, exist_ok=True)

for s in range(S):
    # Get image: (3, H, W) -> (H, W, 3)
    img = images_for_viz[s].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1) * 255.0
    img = img.astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Draw trajectory lines up to current frame
    for i in range(N):
        if not vis_mask[s, i]:
            continue
        
        # Draw trajectory from first frame to current frame
        prev_pt = None
        for prev_s in range(s + 1):
            if vis_mask[prev_s, i]:
                x, y = tracks_for_viz[prev_s, i, 0].item(), tracks_for_viz[prev_s, i, 1].item()
                curr_pt = (int(round(x)), int(round(y)))
                
                # Draw line from previous point to current point
                if prev_pt is not None:
                    R, G, B = track_colors_rgb[i]
                    color_bgr = (int(B), int(G), int(R))
                    cv2.line(img_bgr, prev_pt, curr_pt, color_bgr, thickness=2)
                
                prev_pt = curr_pt
        
        # Draw current point with highlight
        x, y = tracks_for_viz[s, i, 0].item(), tracks_for_viz[s, i, 1].item()
        pt = (int(round(x)), int(round(y)))
        R, G, B = track_colors_rgb[i]
        color_bgr = (int(B), int(G), int(R))
        # Draw filled circle
        cv2.circle(img_bgr, pt, radius=5, color=color_bgr, thickness=-1)
        # Draw white outline
        cv2.circle(img_bgr, pt, radius=5, color=(255, 255, 255), thickness=2)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Save individual frame
    frame_path = os.path.join(output_dir, f"frame_{s:04d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    frame_images.append(img_rgb)

# Create grid image
frames_per_row = 5
num_rows = (S + frames_per_row - 1) // frames_per_row
grid_img = None

for row in range(num_rows):
    start_idx = row * frames_per_row
    end_idx = min(start_idx + frames_per_row, S)
    row_img = np.concatenate(frame_images[start_idx:end_idx], axis=1)
    
    if end_idx - start_idx < frames_per_row:
        padding_width = (frames_per_row - (end_idx - start_idx)) * W
        padding = np.zeros((H, padding_width, 3), dtype=np.uint8)
        row_img = np.concatenate([row_img, padding], axis=1)
    
    if grid_img is None:
        grid_img = row_img
    else:
        grid_img = np.concatenate([grid_img, row_img], axis=0)

grid_path = os.path.join(output_dir, "tracks_grid.png")
cv2.imwrite(grid_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

# 在原始图像上可视化追踪结果
print(f"\nVisualizing tracks on original images...")
original_output_dir = os.path.join(image_folder, "track_visualization_original")
os.makedirs(original_output_dir, exist_ok=True)

original_frame_images = []
for s in range(S):
    # 加载原始图像
    original_img = cv2.imread(image_names[s])
    if original_img is None:
        print(f"Warning: Could not load original image: {image_names[s]}")
        continue
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img_rgb.shape[:2]
    
    # 在原始图像上绘制追踪轨迹
    for i in range(N):
        if not vis_mask[s, i]:
            continue
        
        # 获取当前帧的点坐标
        x_curr, y_curr = tracks_original[s, i, 0].item(), tracks_original[s, i, 1].item()
        curr_pt_valid = (0 <= x_curr < orig_w and 0 <= y_curr < orig_h)
        
        # 绘制从第一帧到当前帧的轨迹
        prev_pt = None
        for prev_s in range(s + 1):
            if vis_mask[prev_s, i]:
                x, y = tracks_original[prev_s, i, 0].item(), tracks_original[prev_s, i, 1].item()
                # 检查坐标是否在图像范围内
                if 0 <= x < orig_w and 0 <= y < orig_h:
                    curr_pt = (int(round(x)), int(round(y)))
                    
                    # 绘制从上一个点到当前点的线
                    if prev_pt is not None:
                        R, G, B = track_colors_rgb[i]
                        color_bgr = (int(B), int(G), int(R))
                        cv2.line(original_img_rgb, prev_pt, curr_pt, color_bgr, thickness=3)
                    
                    prev_pt = curr_pt
        
        # 绘制当前帧的点（如果坐标有效）
        if curr_pt_valid:
            pt = (int(round(x_curr)), int(round(y_curr)))
            R, G, B = track_colors_rgb[i]
            color_bgr = (int(B), int(G), int(R))
            # 绘制填充圆
            cv2.circle(original_img_rgb, pt, radius=8, color=color_bgr, thickness=-1)
            # 绘制白色轮廓
            cv2.circle(original_img_rgb, pt, radius=8, color=(255, 255, 255), thickness=2)
    
    # 保存原始图像上的追踪结果
    original_frame_path = os.path.join(original_output_dir, f"frame_{s:04d}.png")
    cv2.imwrite(original_frame_path, cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))
    original_frame_images.append(original_img_rgb)

# 创建原始图像的网格图
if len(original_frame_images) > 0:
    orig_frames_per_row = 5
    orig_num_rows = (len(original_frame_images) + orig_frames_per_row - 1) // orig_frames_per_row
    orig_grid_img = None
    
    for row in range(orig_num_rows):
        start_idx = row * orig_frames_per_row
        end_idx = min(start_idx + orig_frames_per_row, len(original_frame_images))
        row_img = np.concatenate(original_frame_images[start_idx:end_idx], axis=1)
        
        if end_idx - start_idx < orig_frames_per_row:
            # 获取第一张图像的尺寸
            first_h, first_w = original_frame_images[0].shape[:2]
            padding_width = (orig_frames_per_row - (end_idx - start_idx)) * first_w
            padding = np.zeros((first_h, padding_width, 3), dtype=np.uint8)
            row_img = np.concatenate([row_img, padding], axis=1)
        
        if orig_grid_img is None:
            orig_grid_img = row_img
        else:
            orig_grid_img = np.concatenate([orig_grid_img, row_img], axis=0)
    
    orig_grid_path = os.path.join(original_output_dir, "tracks_grid_original.png")
    cv2.imwrite(orig_grid_path, cv2.cvtColor(orig_grid_img, cv2.COLOR_RGB2BGR))

# 可视化深度图和置信度图
print(f"\nVisualizing depth maps and confidence maps...")
depth_output_dir = os.path.join(image_folder, "depth_visualization")
os.makedirs(depth_output_dir, exist_ok=True)

S_depth = depth_map_np.shape[0]
H_depth, W_depth = depth_map_np.shape[1], depth_map_np.shape[2]

# 计算深度值的范围（排除异常值）
depth_valid = depth_map_np[depth_map_np > 0]
if len(depth_valid) > 0:
    depth_min = np.percentile(depth_valid, 1)
    depth_max = np.percentile(depth_valid, 99)
else:
    depth_min, depth_max = depth_map_np.min(), depth_map_np.max()

# 计算置信度值的范围
conf_min = np.percentile(depth_conf_np[depth_conf_np > 0], 1) if np.any(depth_conf_np > 0) else depth_conf_np.min()
conf_max = np.percentile(depth_conf_np[depth_conf_np > 0], 99) if np.any(depth_conf_np > 0) else depth_conf_np.max()

depth_vis_images = []
conf_vis_images = []

for s in range(S_depth):
    # 可视化深度图
    depth_frame = depth_map_np[s].copy()
    # 归一化到 [0, 1]
    depth_normalized = np.clip((depth_frame - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
    # 应用colormap (使用jet colormap)
    depth_colored = cm.jet(depth_normalized)[:, :, :3]  # (H, W, 3), RGB in [0,1]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # 可视化置信度图
    conf_frame = depth_conf_np[s].copy()
    # 归一化到 [0, 1]
    conf_normalized = np.clip((conf_frame - conf_min) / (conf_max - conf_min + 1e-8), 0, 1)
    # 应用colormap (使用hot colormap)
    conf_colored = cm.hot(conf_normalized)[:, :, :3]  # (H, W, 3), RGB in [0,1]
    conf_colored = (conf_colored * 255).astype(np.uint8)
    
    # 保存单独的深度图和置信度图
    depth_path = os.path.join(depth_output_dir, f"depth_{s:04d}.png")
    conf_path = os.path.join(depth_output_dir, f"confidence_{s:04d}.png")
    cv2.imwrite(depth_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(conf_path, cv2.cvtColor(conf_colored, cv2.COLOR_RGB2BGR))
    
    depth_vis_images.append(depth_colored)
    conf_vis_images.append(conf_colored)

# 创建深度图的网格
depth_frames_per_row = 5
depth_num_rows = (S_depth + depth_frames_per_row - 1) // depth_frames_per_row
depth_grid_img = None

for row in range(depth_num_rows):
    start_idx = row * depth_frames_per_row
    end_idx = min(start_idx + depth_frames_per_row, S_depth)
    row_img = np.concatenate(depth_vis_images[start_idx:end_idx], axis=1)
    
    if end_idx - start_idx < depth_frames_per_row:
        padding_width = (depth_frames_per_row - (end_idx - start_idx)) * W_depth
        padding = np.zeros((H_depth, padding_width, 3), dtype=np.uint8)
        row_img = np.concatenate([row_img, padding], axis=1)
    
    if depth_grid_img is None:
        depth_grid_img = row_img
    else:
        depth_grid_img = np.concatenate([depth_grid_img, row_img], axis=0)

depth_grid_path = os.path.join(depth_output_dir, "depth_grid.png")
cv2.imwrite(depth_grid_path, cv2.cvtColor(depth_grid_img, cv2.COLOR_RGB2BGR))

# 创建置信度图的网格
conf_grid_img = None

for row in range(depth_num_rows):
    start_idx = row * depth_frames_per_row
    end_idx = min(start_idx + depth_frames_per_row, S_depth)
    row_img = np.concatenate(conf_vis_images[start_idx:end_idx], axis=1)
    
    if end_idx - start_idx < depth_frames_per_row:
        padding_width = (depth_frames_per_row - (end_idx - start_idx)) * W_depth
        padding = np.zeros((H_depth, padding_width, 3), dtype=np.uint8)
        row_img = np.concatenate([row_img, padding], axis=1)
    
    if conf_grid_img is None:
        conf_grid_img = row_img
    else:
        conf_grid_img = np.concatenate([conf_grid_img, row_img], axis=0)

conf_grid_path = os.path.join(depth_output_dir, "confidence_grid.png")
cv2.imwrite(conf_grid_path, cv2.cvtColor(conf_grid_img, cv2.COLOR_RGB2BGR))

# 保存原始深度值和置信度值（numpy格式）
np.savez(
    os.path.join(depth_output_dir, "depth_data.npz"),
    depth_map=depth_map_np,
    depth_conf=depth_conf_np,
    depth_min=depth_min,
    depth_max=depth_max,
    conf_min=conf_min,
    conf_max=conf_max
)

print(f"\nVisualization complete!")
print(f"  - Preprocessed frames saved to: {output_dir}/frame_*.png")
print(f"  - Preprocessed grid image saved to: {output_dir}/tracks_grid.png")
print(f"  - Original frames saved to: {original_output_dir}/frame_*.png")
if len(original_frame_images) > 0:
    print(f"  - Original grid image saved to: {original_output_dir}/tracks_grid_original.png")
print(f"  - Depth maps saved to: {depth_output_dir}/depth_*.png")
print(f"  - Depth grid saved to: {depth_output_dir}/depth_grid.png")
print(f"  - Confidence maps saved to: {depth_output_dir}/confidence_*.png")
print(f"  - Confidence grid saved to: {depth_output_dir}/confidence_grid.png")
print(f"  - Raw depth data saved to: {depth_output_dir}/depth_data.npz")
print(f"  - Depth range: [{depth_min:.4f}, {depth_max:.4f}]")
print(f"  - Confidence range: [{conf_min:.4f}, {conf_max:.4f}]")