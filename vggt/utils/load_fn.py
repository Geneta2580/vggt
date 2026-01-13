# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


@dataclass
class CoordinateTransform:
    """
    坐标转换参数，用于在原始图像和预处理图像之间进行双向转换
    """
    orig_width: int
    orig_height: int
    preprocessed_width: int
    preprocessed_height: int
    mode: str  # "crop" or "pad"
    target_size: int
    scale: float  # resize scale factor
    crop_start_y: Optional[int] = None  # crop模式的y偏移
    pad_left: Optional[int] = None  # pad模式的左padding
    pad_top: Optional[int] = None  # pad模式的上padding
    new_width_after_resize: Optional[int] = None  # resize后的宽度（pad模式）
    new_height_after_resize: Optional[int] = None  # resize后的高度（pad模式）
    
    def original_to_preprocessed(self, coords: np.ndarray) -> np.ndarray:
        """
        将原始图像坐标转换为预处理图像坐标
        
        Args:
            coords: (N, 2) array, 原始图像坐标 [x, y]
        
        Returns:
            (N, 2) array, 预处理图像坐标
        """
        coords = coords.copy()
        
        if self.mode == "crop":
            # 应用resize scale
            coords[:, 0] = coords[:, 0] * self.scale
            coords[:, 1] = coords[:, 1] * self.scale
            
            # 如果进行了crop，减去crop偏移
            if self.crop_start_y is not None:
                coords[:, 1] = coords[:, 1] - self.crop_start_y
        else:  # pad mode
            # 应用resize scale
            coords[:, 0] = coords[:, 0] * self.scale
            coords[:, 1] = coords[:, 1] * self.scale
            
            # 应用padding偏移
            if self.pad_left is not None:
                coords[:, 0] = coords[:, 0] + self.pad_left
            if self.pad_top is not None:
                coords[:, 1] = coords[:, 1] + self.pad_top
        
        return coords
    
    def preprocessed_to_original(self, coords: np.ndarray) -> np.ndarray:
        """
        将预处理图像坐标转换为原始图像坐标
        
        Args:
            coords: (N, 2) array, 预处理图像坐标 [x, y]
        
        Returns:
            (N, 2) array, 原始图像坐标
        """
        coords = coords.copy()
        
        if self.mode == "crop":
            # 如果进行了crop，加上crop偏移
            if self.crop_start_y is not None:
                coords[:, 1] = coords[:, 1] + self.crop_start_y
            
            # 反向应用resize scale
            coords[:, 0] = coords[:, 0] / self.scale
            coords[:, 1] = coords[:, 1] / self.scale
        else:  # pad mode
            # 反向应用padding偏移
            if self.pad_left is not None:
                coords[:, 0] = coords[:, 0] - self.pad_left
            if self.pad_top is not None:
                coords[:, 1] = coords[:, 1] - self.pad_top
            
            # 反向应用resize scale
            coords[:, 0] = coords[:, 0] / self.scale
            coords[:, 1] = coords[:, 1] / self.scale
        
        return coords


def load_and_preprocess_images(image_path_list, mode="crop", return_transforms=False):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
        return_transforms (bool, optional): If True, also return coordinate transform objects for each image.
                                          Default: False

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)
        List[CoordinateTransform] (optional): List of coordinate transform objects, one per image.
                                            Only returned if return_transforms=True

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    transforms = []  # 存储每张图像的转换参数
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        # 计算转换参数
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                scale = target_size / width
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
                scale = target_size / height
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
            scale = target_size / width

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # 初始化转换参数
        crop_start_y = None
        pad_left = None
        pad_top = None

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
            crop_start_y = start_y

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        # 创建并存储转换对象
        transform = CoordinateTransform(
            orig_width=width,
            orig_height=height,
            preprocessed_width=img.shape[2],
            preprocessed_height=img.shape[1],
            mode=mode,
            target_size=target_size,
            scale=scale,
            crop_start_y=crop_start_y,
            pad_left=pad_left,
            pad_top=pad_top,
            new_width_after_resize=new_width,
            new_height_after_resize=new_height
        )
        transforms.append(transform)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    if return_transforms:
        return images, transforms
    return images


def convert_original_to_preprocessed_coords(
    original_coords,  # (N, 2) 原始图像像素坐标
    image_path,        # 图像路径，用于获取原始尺寸
    mode="crop",       # 与load_and_preprocess_images的mode一致
    target_size=518
):
    """
    将原始图像的像素坐标转换为预处理后图像的坐标
    
    Args:
        original_coords: torch.Tensor, shape (N, 2), 原始图像的像素坐标 [x, y]
        image_path: str, 图像文件路径
        mode: str, "crop" 或 "pad", 与load_and_preprocess_images的mode一致
        target_size: int, 目标尺寸，默认518
    
    Returns:
        torch.Tensor: shape (N, 2), 预处理后图像的坐标
    """
    # 获取原始图像尺寸
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    orig_width, orig_height = img.size
    
    # 转换为numpy进行计算
    coords = original_coords.cpu().numpy() if isinstance(original_coords, torch.Tensor) else original_coords
    coords = coords.copy()
    
    if mode == "crop":
        # 计算resize后的尺寸（宽度固定为518）
        scale = target_size / orig_width
        new_height = round(orig_height * scale / 14) * 14
        
        # 应用resize scale
        coords[:, 0] = coords[:, 0] * scale  # x坐标
        coords[:, 1] = coords[:, 1] * scale  # y坐标
        
        # 如果高度超过518，进行center crop
        if new_height > target_size:
            crop_start_y = (new_height - target_size) // 2
            coords[:, 1] = coords[:, 1] - crop_start_y
            # 裁剪掉的部分坐标无效，但这里仍然返回（可能需要后续检查边界）
    
    else:  # mode == "pad"
        # 计算resize后的尺寸（最大边为518）
        if orig_width >= orig_height:
            scale = target_size / orig_width
            new_width = target_size
            new_height = round(orig_height * scale / 14) * 14
        else:
            scale = target_size / orig_height
            new_height = target_size
            new_width = round(orig_width * scale / 14) * 14
        
        # 应用resize scale
        coords[:, 0] = coords[:, 0] * scale
        coords[:, 1] = coords[:, 1] * scale
        
        # 计算padding偏移
        h_padding = target_size - new_height
        w_padding = target_size - new_width
        pad_top = h_padding // 2
        pad_left = w_padding // 2
        
        # 应用padding偏移
        coords[:, 0] = coords[:, 0] + pad_left
        coords[:, 1] = coords[:, 1] + pad_top
    
    return torch.FloatTensor(coords)


def convert_preprocessed_to_original_coords(
    preprocessed_coords,  # (N, 2) 预处理后图像的像素坐标
    image_path,           # 图像路径，用于获取原始尺寸
    mode="crop",          # 与load_and_preprocess_images的mode一致
    target_size=518
):
    """
    将预处理后图像的像素坐标转换为原始图像的坐标
    
    Args:
        preprocessed_coords: torch.Tensor, shape (N, 2), 预处理后图像的像素坐标 [x, y]
        image_path: str, 图像文件路径
        mode: str, "crop" 或 "pad", 与load_and_preprocess_images的mode一致
        target_size: int, 目标尺寸，默认518
    
    Returns:
        torch.Tensor: shape (N, 2), 原始图像的坐标
    """
    # 获取原始图像尺寸
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    orig_width, orig_height = img.size
    
    # 转换为numpy进行计算
    coords = preprocessed_coords.cpu().numpy() if isinstance(preprocessed_coords, torch.Tensor) else preprocessed_coords
    coords = coords.copy()
    
    if mode == "crop":
        # 计算resize后的尺寸（宽度固定为518）
        scale = target_size / orig_width
        new_height = round(orig_height * scale / 14) * 14
        
        # 如果高度超过518，进行center crop（反向操作：加上crop偏移）
        if new_height > target_size:
            crop_start_y = (new_height - target_size) // 2
            coords[:, 1] = coords[:, 1] + crop_start_y
        
        # 反向应用resize scale
        coords[:, 0] = coords[:, 0] / scale  # x坐标
        coords[:, 1] = coords[:, 1] / scale  # y坐标
    
    else:  # mode == "pad"
        # 计算resize后的尺寸（最大边为518）
        if orig_width >= orig_height:
            scale = target_size / orig_width
            new_width = target_size
            new_height = round(orig_height * scale / 14) * 14
        else:
            scale = target_size / orig_height
            new_height = target_size
            new_width = round(orig_width * scale / 14) * 14
        
        # 计算padding偏移
        h_padding = target_size - new_height
        w_padding = target_size - new_width
        pad_top = h_padding // 2
        pad_left = w_padding // 2
        
        # 反向应用padding偏移（减去padding）
        coords[:, 0] = coords[:, 0] - pad_left
        coords[:, 1] = coords[:, 1] - pad_top
        
        # 反向应用resize scale
        coords[:, 0] = coords[:, 0] / scale
        coords[:, 1] = coords[:, 1] / scale
    
    return torch.FloatTensor(coords)


def convert_original_to_preprocessed_coords_batch(
    original_coords_list,  # List of (N_i, 2) tensors, 每张图像对应的原始坐标
    image_paths,           # List of image paths
    mode="crop",
    target_size=518
):
    """
    批量转换多张图像的坐标
    
    Args:
        original_coords_list: List[torch.Tensor], 每张图像对应的原始坐标
        image_paths: List[str], 图像路径列表
        mode: str, "crop" 或 "pad"
        target_size: int, 目标尺寸
    
    Returns:
        List[torch.Tensor]: 转换后的坐标列表
    """
    converted_coords = []
    for coords, img_path in zip(original_coords_list, image_paths):
        converted = convert_original_to_preprocessed_coords(
            coords, img_path, mode, target_size
        )
        converted_coords.append(converted)
    return converted_coords


def convert_preprocessed_to_original_coords_batch(
    preprocessed_coords_list,  # List of (N_i, 2) tensors, 每张图像对应的预处理后坐标
    image_paths,                # List of image paths
    mode="crop",
    target_size=518
):
    """
    批量将预处理后的坐标转换回原始图像坐标
    
    Args:
        preprocessed_coords_list: List[torch.Tensor], 每张图像对应的预处理后坐标
        image_paths: List[str], 图像路径列表
        mode: str, "crop" 或 "pad"
        target_size: int, 目标尺寸
    
    Returns:
        List[torch.Tensor]: 转换后的原始图像坐标列表
    """
    converted_coords = []
    for coords, img_path in zip(preprocessed_coords_list, image_paths):
        converted = convert_preprocessed_to_original_coords(
            coords, img_path, mode, target_size
        )
        converted_coords.append(converted)
    return converted_coords


def convert_coords_with_transform(
    coords: torch.Tensor,
    transform: CoordinateTransform,
    to_preprocessed: bool = True
) -> torch.Tensor:
    """
    使用预计算的转换对象进行坐标转换（推荐使用，更高效）
    
    Args:
        coords: torch.Tensor, shape (N, 2), 坐标 [x, y]
        transform: CoordinateTransform, 转换对象（从load_and_preprocess_images获取）
        to_preprocessed: bool, True表示原始->预处理，False表示预处理->原始
    
    Returns:
        torch.Tensor: shape (N, 2), 转换后的坐标
    
    Example:
        >>> images, transforms = load_and_preprocess_images(image_paths, return_transforms=True)
        >>> original_coords = torch.FloatTensor([[100, 200]])
        >>> preprocessed_coords = convert_coords_with_transform(original_coords, transforms[0], to_preprocessed=True)
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    
    if to_preprocessed:
        converted = transform.original_to_preprocessed(coords_np)
    else:
        converted = transform.preprocessed_to_original(coords_np)
    
    return torch.FloatTensor(converted)


def convert_coords_batch_with_transforms(
    coords_list: List[torch.Tensor],
    transforms: List[CoordinateTransform],
    to_preprocessed: bool = True
) -> List[torch.Tensor]:
    """
    批量使用转换对象进行坐标转换
    
    Args:
        coords_list: List[torch.Tensor], 每张图像对应的坐标列表
        transforms: List[CoordinateTransform], 转换对象列表（从load_and_preprocess_images获取）
        to_preprocessed: bool, True表示原始->预处理，False表示预处理->原始
    
    Returns:
        List[torch.Tensor]: 转换后的坐标列表
    """
    converted_list = []
    for coords, transform in zip(coords_list, transforms):
        converted = convert_coords_with_transform(coords, transform, to_preprocessed)
        converted_list.append(converted)
    return converted_list
