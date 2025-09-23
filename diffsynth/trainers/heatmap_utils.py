"""
Heatmap to Colormap conversion utilities for Wan2.2 training
基于reconstruct_heatmap/test_heatmap_peak_accuracy.py的热力图转换工具
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Union, Tuple
from matplotlib import cm


def convert_heatmap_to_colormap(heatmap, colormap_name='viridis'):
    """
    Convert heatmap to RGB image using matplotlib colormap (optimized)
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Normalize heatmap to [0, 1] if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Apply colormap (vectorized operation)
    colormap = cm.get_cmap(colormap_name)
    rgb_image = colormap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    return rgb_image.astype(np.float32)


def extract_heatmap_from_colormap(rgb_image, colormap_name='viridis'):
    """
    Extract heatmap from RGB colormap image by finding closest colormap values (adaptive)
    基于test_heatmap_peak_accuracy.py中的实现
    """
    h, w = rgb_image.shape[:2]
    colormap = cm.get_cmap(colormap_name)

    # Adaptive algorithm selection based on image size
    total_pixels = h * w
    use_vectorized = total_pixels > 16384  # 128x128 threshold

    if use_vectorized:
        # Vectorized method for large images - optimized to 128 points for speed
        reference_values = np.linspace(0, 1, 128)
        reference_colors = colormap(reference_values)[:, :3]

        rgb_flat = rgb_image.reshape(-1, 3)
        distances = np.sum((rgb_flat[:, None, :] - reference_colors[None, :, :]) ** 2, axis=2)
        closest_indices = np.argmin(distances, axis=1)
        extracted_values = reference_values[closest_indices]
        extracted_heatmap = extracted_values.reshape(h, w)

    else:
        # Loop method for small images - also optimized to 128 points
        reference_values = np.linspace(0, 1, 128)
        reference_colors = colormap(reference_values)[:, :3]

        extracted_heatmap = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                pixel = rgb_image[i, j]
                distances = np.sum((reference_colors - pixel) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                extracted_heatmap[i, j] = reference_values[closest_idx]

    return extracted_heatmap.astype(np.float32)


def convert_color_to_wan_format(image):
    """
    Convert color image to Wan-VAE format
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Convert to torch tensor and permute dimensions
    # Shape: (H, W, 3) -> (3, H, W) -> (1, 3, 1, H, W)
    image_tensor = torch.from_numpy(image).float()
    image_chw = image_tensor.permute(2, 0, 1)  # H,W,C -> C,H,W
    image_5d = image_chw.unsqueeze(0).unsqueeze(2)  # Add batch and time dimensions

    return image_5d


def convert_from_wan_format(decoded_5d):
    """
    Convert decoded output back to image format
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Shape: (1, 3, 1, H, W) -> (3, H, W) -> (H, W, 3)
    decoded_chw = decoded_5d.squeeze(0).squeeze(1)  # Remove batch and time dims
    decoded_hwc = decoded_chw.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C

    # Clamp to [0, 1] range
    decoded_hwc = np.clip(decoded_hwc, 0, 1)

    return decoded_hwc


def heatmap_sequence_to_pil_images(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                                 colormap_name: str = 'viridis') -> List[Image.Image]:
    """
    将热力图序列转换为PIL图像列表，用于Wan2.2训练

    Args:
        heatmap_sequence: 热力图序列 (T, H, W)
        colormap_name: colormap名称

    Returns:
        PIL图像列表
    """
    # 转换为numpy
    if isinstance(heatmap_sequence, torch.Tensor):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    pil_images = []
    for t in range(heatmap_sequence.shape[0]):
        heatmap = heatmap_sequence[t]  # (H, W)

        # 转换为colormap
        colormap_image = convert_heatmap_to_colormap(heatmap, colormap_name)

        # 转换为PIL图像 (需要将值范围调整到[0, 255])
        colormap_uint8 = (colormap_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(colormap_uint8)
        pil_images.append(pil_image)

    return pil_images


def rgb_tensor_to_pil_image(rgb_tensor: torch.Tensor) -> Image.Image:
    """
    将RGB tensor转换为PIL图像

    Args:
        rgb_tensor: RGB tensor (3, H, W) 或 (H, W, 3)

    Returns:
        PIL图像
    """
    # 转换为numpy
    rgb_array = rgb_tensor.cpu().numpy()

    # 处理通道顺序
    if rgb_array.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        rgb_array = rgb_array.transpose(1, 2, 0)

    # 确保数值范围正确
    if rgb_array.max() <= 1.0:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    else:
        rgb_array = rgb_array.astype(np.uint8)

    return Image.fromarray(rgb_array)


def prepare_heatmap_data_for_wan(rgb_image: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'viridis') -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式

    Args:
        rgb_image: RGB图像 tensor (3, H, W)
        heatmap_sequence: 热力图序列 tensor (T, H, W)
        instruction: 文本指令
        colormap_name: colormap名称

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
    """
    # 1. 转换RGB图像为PIL
    first_frame = rgb_tensor_to_pil_image(rgb_image)

    # 2. 转换热力图序列为PIL图像列表
    video_frames = heatmap_sequence_to_pil_images(heatmap_sequence, colormap_name)

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,
        'input_image': first_frame  # 首帧作为条件输入
    }

    return data


# 测试函数
if __name__ == "__main__":
    print("Testing heatmap conversion utilities...")

    # 创建测试数据
    test_heatmap = np.random.rand(64, 64)
    test_sequence = np.random.rand(5, 64, 64)

    # 测试单帧转换
    print("Testing single heatmap conversion...")
    colormap_img = convert_heatmap_to_colormap(test_heatmap)
    restored_heatmap = extract_heatmap_from_colormap(colormap_img)
    print(f"Original shape: {test_heatmap.shape}, Restored shape: {restored_heatmap.shape}")

    # 测试序列转换
    print("Testing sequence conversion...")
    pil_images = heatmap_sequence_to_pil_images(test_sequence)
    print(f"Generated {len(pil_images)} PIL images")

    # 测试完整数据准备
    print("Testing complete data preparation...")
    rgb_tensor = torch.rand(3, 64, 64)
    heatmap_tensor = torch.rand(5, 64, 64)
    data = prepare_heatmap_data_for_wan(rgb_tensor, heatmap_tensor, "test instruction")
    print(f"Data keys: {list(data.keys())}")
    print(f"Video frames: {len(data['video'])}")

    print("All tests completed successfully!")