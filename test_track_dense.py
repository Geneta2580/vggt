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
# image_folder = "/home/geneta/dataset/source_dataset/euroc/MH_05_difficult/MH05_dark"  # 修改为你的图片文件夹路径
image_folder = "/home/geneta/dataset/source_dataset/euroc/V1_03_difficult/V103_change_pair"  # 修改为你的图片文件夹路径


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

# 读取图像到列表中
print("\nReading images from disk...")
read_start_time = time.time()
image_list = []
for img_path in image_names:
    # 使用 PIL Image 读取图像
    img = Image.open(img_path)
    image_list.append(img)
read_time = time.time() - read_start_time
print(f"Image reading time: {read_time:.3f}s")

# 加载图像并获取坐标转换对象（使用双向映射）
print("\nLoading and preprocessing images...")
load_start_time = time.time()
images, transforms = load_and_preprocess_images(image_list, mode="crop", return_transforms=True)
load_time = time.time() - load_start_time
print(f"Image loading time: {load_time:.3f}s")

images = images.to(device)

# Convert from (N, C, H, W) to (1, N, C, H, W) where N is sequence length
# aggregator expects shape [B, S, C, H, W] where B=batch, S=sequence
if len(images.shape) == 4:
    images = images.unsqueeze(0)  # Add batch dimension

print(f"Images shape: {images.shape}")  # Should be (1, N, 3, H, W)

# 开始推理计时
print("\n" + "="*60)
print("Starting inference (Round 1 - Warmup)...")
print("="*60)

# 第一轮推理（Warmup）
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        pose_enc_list = model.camera_head(aggregated_tokens_list)

print("\n" + "="*60)
print("Starting inference (Round 2 - Actual Measurement)...")
print("="*60)
inference_start_time = time.time()

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Aggregator阶段
        agg_start = time.time()
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        agg_time = time.time() - agg_start
        print(f"Aggregator time: {agg_time:.3f}s")

        # Camera Head阶段
        camera_start = time.time()
        pose_enc_list = model.camera_head(aggregated_tokens_list)
        camera_time = time.time() - camera_start
        print(f"Camera head time: {camera_time:.3f}s")

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
        
        # 使用OpenCV的goodFeaturesToTrack从第一帧原始图像提取角点
        print("\nExtracting corner points from first frame using OpenCV goodFeaturesToTrack...")
        
        # 加载第一帧原始图像
        first_frame_original = cv2.imread(image_names[0])
        if first_frame_original is None:
            raise ValueError(f"Could not load first frame: {image_names[0]}")
        
        # 转换为灰度图
        first_frame_gray = cv2.cvtColor(first_frame_original, cv2.COLOR_BGR2GRAY)
        
        # 使用goodFeaturesToTrack提取角点
        max_corners = 600
        quality_level = 0.005
        min_distance = 10
        block_size = 7
        use_harris = False
        k = 0.04
        
        corners = cv2.goodFeaturesToTrack(
            first_frame_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            useHarrisDetector=use_harris,
            k=k
        )
        
        if corners is None or len(corners) == 0:
            raise ValueError("No corners detected in first frame")
        
        # corners shape is (N, 1, 2), 转换为 (N, 2)
        corners = corners.reshape(-1, 2)  # (N, 2) 格式，坐标是 [x, y]
        
        print(f"Detected {len(corners)} corner points from first frame")
        
        # 将角点坐标从原始图像空间转换到预处理后的图像空间
        corners_tensor_original = torch.FloatTensor(corners)  # (N, 2)
        selected_points_tensor = convert_coords_with_transform(
            corners_tensor_original,
            transforms[0],  # 使用第一帧的转换对象
            to_preprocessed=True
        ).to(device)  # (N, 2)
        
        # 保存原始角点坐标（用于可视化）
        selected_points = corners  # 原始图像坐标
        selected_points_preprocessed = selected_points_tensor.cpu().numpy()  # 预处理后坐标
        
        print(f"Converted {len(selected_points_tensor)} points to preprocessed image space")
        print(f"Original image size: {first_frame_gray.shape[1]}x{first_frame_gray.shape[0]}")
        print(f"Preprocessed image size: {images.shape[3]}x{images.shape[4]}")
        
        # 追踪预测阶段
        track_start = time.time()
        track, vis_score, conf_score = model.track_head(
            aggregated_tokens_list, images, ps_idx, query_points=selected_points_tensor[None]
        )
        track_time = time.time() - track_start
        print(f"Tracking time: {track_time:.3f}s")

# 总推理时间
total_inference_time = time.time() - inference_start_time
print("="*60)
print(f"Total inference time: {total_inference_time:.3f}s")
print(f"  - Aggregator: {agg_time:.3f}s ({agg_time/total_inference_time*100:.1f}%)")
print(f"  - Camera head: {camera_time:.3f}s ({camera_time/total_inference_time*100:.1f}%)")
print(f"  - Depth prediction: {depth_time:.3f}s ({depth_time/total_inference_time*100:.1f}%)")
print(f"  - Tracking: {track_time:.3f}s ({track_time/total_inference_time*100:.1f}%)")
print(f"Average time per frame: {total_inference_time/len(image_names):.3f}s")
print("="*60)

print(f"\nPose encoding shape: {pose_enc_list[-1].shape}")
print(f"Depth map shape: {depth_map.shape}")
print(f"Depth confidence shape: {depth_conf.shape}")
print(f"Track shape: {track.shape}")
print(f"Visibility scores shape: {vis_score.shape}")
print(f"Confidence scores shape: {conf_score.shape}")
print(f"Number of tracked points: {track.shape[2]}")

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
output_dir = os.path.join(image_folder, "track_visualization_dense")
os.makedirs(output_dir, exist_ok=True)

# Enhanced visualization with trajectory lines
S, N, _ = tracks_for_viz.shape
H, W = images_for_viz.shape[2], images_for_viz.shape[3]

print(f"\nVisualizing {N} tracked points and saving to: {output_dir}")

# 在第一帧上可视化提取的角点（预处理后的图像空间）
print("\nVisualizing extracted corner points on first frame (preprocessed)...")
first_frame_img = images_for_viz[0].permute(1, 2, 0).numpy()
first_frame_img = np.clip(first_frame_img, 0, 1) * 255.0
first_frame_img = first_frame_img.astype(np.uint8)
first_frame_bgr = cv2.cvtColor(first_frame_img, cv2.COLOR_RGB2BGR)

# 在预处理后的第一帧上绘制提取的角点
for i, (x, y) in enumerate(selected_points_preprocessed):
    pt = (int(round(x)), int(round(y)))
    # 检查坐标是否在图像范围内
    if 0 <= pt[0] < W and 0 <= pt[1] < H:
        # 使用绿色标记角点
        color_bgr = (0, 255, 0)  # 绿色
        # 绘制填充圆
        cv2.circle(first_frame_bgr, pt, radius=3, color=color_bgr, thickness=-1)
        # 绘制白色轮廓
        cv2.circle(first_frame_bgr, pt, radius=3, color=(255, 255, 255), thickness=1)

# 保存第一帧的选中点可视化
first_frame_path = os.path.join(output_dir, "selected_points_first_frame.png")
cv2.imwrite(first_frame_path, first_frame_bgr)
print(f"Saved selected points visualization to: {first_frame_path}")

# 在原始第一帧上可视化提取的角点
print("\nVisualizing extracted corner points on original first frame...")
first_frame_original_viz = first_frame_original.copy()
for i, (x, y) in enumerate(selected_points):
    pt = (int(round(x)), int(round(y)))
    # 检查坐标是否在图像范围内
    orig_h, orig_w = first_frame_original_viz.shape[:2]
    if 0 <= pt[0] < orig_w and 0 <= pt[1] < orig_h:
        # 使用绿色标记角点
        color_bgr = (0, 255, 0)  # 绿色
        # 绘制填充圆
        cv2.circle(first_frame_original_viz, pt, radius=5, color=color_bgr, thickness=-1)
        # 绘制白色轮廓
        cv2.circle(first_frame_original_viz, pt, radius=5, color=(255, 255, 255), thickness=2)

# 保存原始第一帧的角点可视化
first_frame_original_path = os.path.join(output_dir, "selected_points_first_frame_original.png")
cv2.imwrite(first_frame_original_path, first_frame_original_viz)
print(f"Saved original first frame corner points visualization to: {first_frame_original_path}")

# 创建置信度热力图（显示第一帧的置信度分布和提取的角点）
# 归一化第一帧的完整置信度图
first_frame_conf = depth_conf_np[0]  # (H, W) - 第一帧的置信度图
conf_full_normalized = (first_frame_conf - first_frame_conf.min()) / (first_frame_conf.max() - first_frame_conf.min() + 1e-8)
# 应用colormap，cm.jet返回(H, W, 4)的RGBA，我们只需要RGB
conf_full_heatmap = cm.jet(conf_full_normalized)[:, :, :3]  # (H, W, 3)
conf_full_heatmap = (conf_full_heatmap * 255).astype(np.uint8)
conf_full_heatmap_bgr = cv2.cvtColor(conf_full_heatmap, cv2.COLOR_RGB2BGR)

# 在置信度热力图上标记提取的角点（使用预处理后的坐标）
for i, (x, y) in enumerate(selected_points_preprocessed):
    pt = (int(round(x)), int(round(y)))
    # 检查坐标是否在图像范围内
    if 0 <= pt[0] < W and 0 <= pt[1] < H:
        cv2.circle(conf_full_heatmap_bgr, pt, radius=2, color=(255, 255, 255), thickness=-1)
        cv2.circle(conf_full_heatmap_bgr, pt, radius=2, color=(0, 0, 0), thickness=1)

conf_heatmap_path = os.path.join(output_dir, "confidence_heatmap_with_corner_points.png")
cv2.imwrite(conf_heatmap_path, conf_full_heatmap_bgr)
print(f"Saved confidence heatmap with corner points to: {conf_heatmap_path}")

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

# 统计追踪信息
print(f"\nTracking statistics:")
print(f"  - Total points tracked: {N}")
print(f"  - Average visibility across all frames: {vis_mask.float().mean():.2%}")
print(f"  - Points visible in first frame: {vis_mask[0].sum().item()}/{N}")
print(f"  - Points visible in last frame: {vis_mask[-1].sum().item()}/{N}")

# 计算每个点的平均可见性
point_visibility = vis_mask.float().mean(dim=0)  # (N,)
print(f"  - Points with >50% visibility: {(point_visibility > 0.5).sum().item()}/{N}")
print(f"  - Points with >80% visibility: {(point_visibility > 0.8).sum().item()}/{N}")
print(f"  - Points with 100% visibility: {(point_visibility == 1.0).sum().item()}/{N}")

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