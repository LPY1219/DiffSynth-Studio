"""
Heatmap Inference Script for Wan2.2
用于热力图序列预测的推断脚本
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Any
import cv2
from pathlib import Path

# 设置环境变量
os.environ["COPPELIASIM_ROOT"] = "/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["COPPELIASIM_ROOT"]
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
os.environ["DISPLAY"] = ":1.0"

# 添加项目路径
sys.path.append("/share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio")
sys.path.append("/share/project/lpy/BridgeVLA/Wan/single_view")

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap

# 导入训练时的数据集工厂
try:
    from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory
    from data.dataset import RobotTrajectoryDataset, ProjectionInterface
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dataset imports failed: {e}")
    DATASET_AVAILABLE = False


class HeatmapInference:
    """热力图推断类"""

    def __init__(self,
                 lora_checkpoint_path: str,
                 model_base_path: str = "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B",
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16):
        """
        初始化推断器

        Args:
            lora_checkpoint_path: LoRA模型检查点路径
            model_base_path: 基础模型路径
            device: 设备
            torch_dtype: 张量类型
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_checkpoint_path = lora_checkpoint_path

        print("Loading Wan2.2 pipeline...")
        # 加载pipeline
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(path=[
                    f"{model_base_path}/diffusion_pytorch_model-00001-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00002-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00003-of-00003.safetensors"
                ]),
                ModelConfig(path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(path=f"{model_base_path}/Wan2.2_VAE.pth"),
            ],
        )

        # 使用pipeline的load_lora方法加载LoRA权重
        print(f"Loading LoRA checkpoint: {lora_checkpoint_path}")
        self._load_lora_weights_pipeline_method()

        print("Pipeline initialized successfully!")

    def _load_lora_weights(self):
        """加载LoRA权重"""
        if not os.path.exists(self.lora_checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {self.lora_checkpoint_path}")

        print(f"Loading LoRA checkpoint from: {self.lora_checkpoint_path}")

        # 加载LoRA状态字典
        lora_state_dict = load_state_dict(self.lora_checkpoint_path)
        print(f"LoRA checkpoint contains {len(lora_state_dict)} parameters")

        # 统计LoRA参数
        lora_a_count = sum(1 for k in lora_state_dict.keys() if 'lora_A' in k)
        lora_b_count = sum(1 for k in lora_state_dict.keys() if 'lora_B' in k)
        print(f"Found {lora_a_count} LoRA A weights and {lora_b_count} LoRA B weights")

        # 检查键名格式并进行必要的映射
        mapped_state_dict = {}
        for key, value in lora_state_dict.items():
            if "lora_A" in key or "lora_B" in key:
                # 如果键名缺少 .default，则添加
                if "lora_A.weight" in key:
                    new_key = key.replace("lora_A.weight", "lora_A.default.weight")
                elif "lora_B.weight" in key:
                    new_key = key.replace("lora_B.weight", "lora_B.default.weight")
                else:
                    # 键名已经是正确格式
                    new_key = key

                mapped_state_dict[new_key] = value

        print(f"Mapped to {len(mapped_state_dict)} parameters for loading")

        # 检查模型中的LoRA模块
        dit_state_dict = self.pipe.dit.state_dict()
        model_lora_keys = [k for k in dit_state_dict.keys() if 'lora_A' in k or 'lora_B' in k]
        print(f"Model has {len(model_lora_keys)} LoRA parameters")

        # 验证键名匹配
        matched_keys = []
        missing_keys = []
        unexpected_keys = []

        for key in mapped_state_dict.keys():
            if key in dit_state_dict:
                matched_keys.append(key)
            else:
                missing_keys.append(key)

        for key in model_lora_keys:
            if key not in mapped_state_dict:
                unexpected_keys.append(key)

        print(f"Key matching results:")
        print(f"  - Matched keys: {len(matched_keys)}")
        print(f"  - Missing keys in checkpoint: {len(unexpected_keys)}")
        print(f"  - Unexpected keys in checkpoint: {len(missing_keys)}")

        if len(matched_keys) == 0:
            print("ERROR: No LoRA keys matched! Checking key format mismatch...")
            # 显示前几个键名以便调试
            print("First 5 checkpoint keys:")
            for i, key in enumerate(list(mapped_state_dict.keys())[:5]):
                print(f"  {key}")
            print("First 5 model LoRA keys:")
            for i, key in enumerate(model_lora_keys[:5]):
                print(f"  {key}")
            raise RuntimeError("Failed to match any LoRA keys")

        # 加载到DIT模型
        load_result = self.pipe.dit.load_state_dict(mapped_state_dict, strict=False)

        print(f"LoRA loading completed:")
        print(f"  - Successfully loaded: {len(matched_keys)} parameters")
        print(f"  - Missing keys: {len(load_result[0])}")
        print(f"  - Unexpected keys: {len(load_result[1])}")

        if len(load_result[0]) > 0:
            print(f"Missing keys (first 5): {load_result[0][:5]}")
        if len(load_result[1]) > 0:
            print(f"Unexpected keys (first 5): {load_result[1][:5]}")

        # 验证LoRA权重确实被加载
        self._verify_lora_loading(matched_keys[:5])

    def _verify_lora_loading(self, sample_keys):
        """验证LoRA权重是否正确加载"""
        if not sample_keys:
            return

        print("Verifying LoRA weight loading:")
        dit_state_dict = self.pipe.dit.state_dict()

        for key in sample_keys:
            if key in dit_state_dict:
                tensor = dit_state_dict[key]
                # 检查张量不全为零（说明确实加载了权重）
                is_zero = torch.allclose(tensor, torch.zeros_like(tensor))
                norm = tensor.norm().item()
                print(f"  {key}: norm={norm:.6f}, all_zero={is_zero}")
            else:
                print(f"  {key}: NOT FOUND in model state dict")

    def _load_lora_weights_pipeline_method(self):
        """使用pipeline内置方法加载LoRA权重"""
        if not os.path.exists(self.lora_checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {self.lora_checkpoint_path}")

        print(f"Loading LoRA using pipeline method from: {self.lora_checkpoint_path}")

        try:
            # 使用pipeline的load_lora方法加载到DIT模型
            self.pipe.load_lora(self.pipe.dit, self.lora_checkpoint_path, alpha=1.0)
            print("LoRA loaded successfully using pipeline method!")

            # 验证加载结果 - 检查DIT模块中的一些关键层
            print("Verifying DIT model structure:")
            sample_modules = []
            for name, module in self.pipe.dit.named_modules():
                if 'cross_attn' in name and ('q' in name or 'k' in name or 'v' in name or 'o' in name):
                    sample_modules.append((name, module))
                    if len(sample_modules) >= 5:
                        break

            for name, module in sample_modules:
                if hasattr(module, 'weight'):
                    weight_norm = module.weight.norm().item()
                    print(f"  {name}: weight norm = {weight_norm:.6f}")

        except Exception as e:
            print(f"Failed to load LoRA using pipeline method: {e}")
            print("Falling back to manual loading method...")
            self._load_lora_weights()

    def predict_heatmap_sequence(self,
                                input_image: Image.Image,
                                prompt: str,
                                num_frames: int = 5,
                                height: int = 256,
                                width: int = 256,
                                seed: int = None) -> List[Image.Image]:
        """
        预测热力图序列

        Args:
            input_image: 输入图像
            prompt: 语言指令
            num_frames: 预测帧数
            height: 输出高度
            width: 输出宽度
            seed: 随机种子

        Returns:
            预测的热力图序列
        """
        print(f"Predicting heatmap sequence...")
        print(f"  Input image size: {input_image.size}")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {num_frames} frames of {width}x{height}")

        # 调整输入图像尺寸
        input_image_resized = input_image.resize((width, height))

        # 生成视频序列
        video_frames = self.pipe(
            prompt=prompt,
            input_image=input_image_resized,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        print(f"Generated {len(video_frames)} frames")
        return video_frames

    def find_peak_position(self, heatmap_image: Image.Image, colormap_name: str = 'viridis') -> Tuple[int, int]:
        """
        找到热力图中的峰值位置

        Args:
            heatmap_image: 热力图PIL图像（可能是colormap格式）
            colormap_name: 使用的colormap名称

        Returns:
            (x, y) 峰值位置坐标
        """
        # 将PIL图像转换为numpy数组
        rgb_array = np.array(heatmap_image)

        # 检查是否是RGB格式（colormap格式）
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            # 如果是RGB格式，从colormap提取热力图数值
            rgb_normalized = rgb_array.astype(np.float32) / 255.0
            heatmap_array = extract_heatmap_from_colormap(rgb_normalized, colormap_name)
        else:
            # 如果是灰度图，直接使用
            heatmap_array = np.array(heatmap_image.convert('L')).astype(np.float32) / 255.0

        # 找到最大值位置
        max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)

        # 返回(x, y)格式，注意numpy是(row, col)格式，需要转换
        return (max_pos[1], max_pos[0])

    def calculate_peak_distance(self, pred_peak: Tuple[int, int], gt_peak: Tuple[int, int]) -> float:
        """
        计算两个峰值位置之间的欧几里得距离

        Args:
            pred_peak: 预测峰值位置
            gt_peak: 真实峰值位置

        Returns:
            欧几里得距离
        """
        return np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    def convert_colormap_to_heatmap(self, colormap_image: Image.Image, colormap_name: str = 'viridis') -> np.ndarray:
        """
        将colormap格式的图像转换为热力图数值数组

        Args:
            colormap_image: colormap格式的PIL图像
            colormap_name: 使用的colormap名称

        Returns:
            热力图数值数组 (H, W)
        """
        # 转换为numpy数组并归一化
        rgb_array = np.array(colormap_image).astype(np.float32) / 255.0

        # 从colormap提取热力图数值
        heatmap_array = extract_heatmap_from_colormap(rgb_array, colormap_name)

        return heatmap_array

    def save_heatmap_visualization(self, output_path: str, original_image: Image.Image,
                                 colormap_sequence: List[Image.Image], colormap_name: str = 'viridis'):
        """
        保存热力图可视化结果，包括原始colormap和提取的热力图

        Args:
            output_path: 输出路径前缀
            original_image: 原始输入图像
            colormap_sequence: colormap格式的预测序列
            colormap_name: 使用的colormap名称
        """
        import matplotlib.pyplot as plt

        num_frames = len(colormap_sequence)
        fig, axes = plt.subplots(3, num_frames + 1, figsize=(3*(num_frames + 1), 9))

        # 如果只有一帧，调整axes格式
        if num_frames == 0:
            return
        if num_frames == 1:
            axes = axes.reshape(3, -1)

        # 第一列：输入图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        # 其余列：预测序列
        for i, colormap_img in enumerate(colormap_sequence):
            col_idx = i + 1

            # 第一行：colormap格式（模型直接输出）
            axes[0, col_idx].imshow(colormap_img)
            axes[0, col_idx].set_title(f'Colormap Frame {i}')
            axes[0, col_idx].axis('off')

            # 第二行：提取的热力图
            heatmap_array = self.convert_colormap_to_heatmap(colormap_img, colormap_name)
            im = axes[1, col_idx].imshow(heatmap_array, cmap='viridis')
            axes[1, col_idx].set_title(f'Extracted Heatmap {i}')
            axes[1, col_idx].axis('off')

            # 找到峰值位置并标记
            peak_pos = self.find_peak_position(colormap_img, colormap_name)
            axes[1, col_idx].plot(peak_pos[0], peak_pos[1], 'r*', markersize=10)

            # 第三行：热力图数值统计
            max_val = np.max(heatmap_array)
            min_val = np.min(heatmap_array)
            mean_val = np.mean(heatmap_array)
            axes[2, col_idx].text(0.1, 0.8, f'Max: {max_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.6, f'Min: {min_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.4, f'Mean: {mean_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.2, f'Peak: ({peak_pos[0]}, {peak_pos[1]})', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].set_title(f'Stats Frame {i}')
            axes[2, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Heatmap visualization saved to: {output_path}")


def test_on_dataset(inference_engine: HeatmapInference,
                   data_root: str,
                   output_dir: str = "./inference_results",
                   num_test_samples: int = 10,
                   sequence_length: int = 5):
    """
    在数据集上进行测试

    Args:
        inference_engine: 推断引擎
        data_root: 数据根目录
        output_dir: 输出目录
        num_test_samples: 测试样本数量
        sequence_length: 序列长度
    """
    if not DATASET_AVAILABLE:
        print("Dataset not available, skipping dataset test")
        return

    print(f"Testing on dataset: {data_root}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建测试数据集
    try:
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
            data_root=data_root,
            sequence_length=sequence_length,
            step_interval=1,
            min_trail_length=10,
            image_size=(256, 256),
            sigma=1.5,
            augmentation=False,  # 测试时不使用数据增强
            mode="train",  # 暂时使用训练集
            scene_bounds=[0, -0.45, -0.05, 0.8, 0.55, 0.6],
            transform_augmentation_xyz=[0.0, 0.0, 0.0],  # 测试时不增强
            transform_augmentation_rpy=[0.0, 0.0, 0.0],
            debug=False,
            colormap_name="viridis",
            repeat=1
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 测试样本
    num_test_samples = min(num_test_samples, len(dataset))
    all_distances = []

    for i in range(num_test_samples):
        print(f"\nProcessing sample {i+1}/{num_test_samples}")

        try:
            # 获取数据样本
            sample = dataset[i]
            input_image = sample['input_image']
            prompt = sample['prompt']
            gt_video = sample['video']  # 真实的热力图序列

            print(f"  Prompt: {prompt[:50]}...")

            # 生成预测
            pred_video = inference_engine.predict_heatmap_sequence(
                input_image=input_image,
                prompt=prompt,
                num_frames=len(gt_video),
                seed=42
            )

            # 计算每帧的峰值距离
            frame_distances = []

            # 创建可视化
            num_frames = len(gt_video)
            fig, axes = plt.subplots(3, num_frames, figsize=(3*num_frames, 9))
            if num_frames == 1:
                axes = axes.reshape(-1, 1)

            for frame_idx in range(num_frames):
                gt_frame = gt_video[frame_idx]
                pred_frame = pred_video[frame_idx] if frame_idx < len(pred_video) else gt_frame

                # 找到峰值位置
                gt_peak = inference_engine.find_peak_position(gt_frame)
                pred_peak = inference_engine.find_peak_position(pred_frame)

                # 计算距离
                distance = inference_engine.calculate_peak_distance(pred_peak, gt_peak)
                frame_distances.append(distance)

                # 可视化
                # 第一行：真实热力图
                axes[0, frame_idx].imshow(gt_frame)
                axes[0, frame_idx].plot(gt_peak[0], gt_peak[1], 'r*', markersize=10, label='GT Peak')
                axes[0, frame_idx].set_title(f'GT Frame {frame_idx}')
                axes[0, frame_idx].legend()
                axes[0, frame_idx].axis('off')

                # 第二行：预测热力图
                axes[1, frame_idx].imshow(pred_frame)
                axes[1, frame_idx].plot(pred_peak[0], pred_peak[1], 'b*', markersize=10, label='Pred Peak')
                axes[1, frame_idx].set_title(f'Pred Frame {frame_idx}')
                axes[1, frame_idx].legend()
                axes[1, frame_idx].axis('off')

                # 第三行：叠加比较
                axes[2, frame_idx].imshow(gt_frame, alpha=0.7)
                axes[2, frame_idx].imshow(pred_frame, alpha=0.3)
                axes[2, frame_idx].plot(gt_peak[0], gt_peak[1], 'r*', markersize=10, label='GT Peak')
                axes[2, frame_idx].plot(pred_peak[0], pred_peak[1], 'b*', markersize=10, label='Pred Peak')
                axes[2, frame_idx].set_title(f'Overlay (Dist: {distance:.1f})')
                axes[2, frame_idx].legend()
                axes[2, frame_idx].axis('off')

            # 添加输入图像
            fig.suptitle(f'Sample {i+1}: {prompt[:50]}...\nAvg Distance: {np.mean(frame_distances):.2f} pixels', fontsize=10)

            # 保存结果
            result_path = os.path.join(output_dir, f'sample_{i+1:03d}_comparison.png')
            plt.tight_layout()
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 保存输入图像
            input_path = os.path.join(output_dir, f'sample_{i+1:03d}_input.png')
            input_image.save(input_path)

            all_distances.extend(frame_distances)
            print(f"  Avg distance for this sample: {np.mean(frame_distances):.2f} pixels")

        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            continue

    # 计算总体统计
    if all_distances:
        avg_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Total frames evaluated: {len(all_distances)}")
        print(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels")
        print(f"Min distance: {np.min(all_distances):.2f} pixels")
        print(f"Max distance: {np.max(all_distances):.2f} pixels")
        print(f"Results saved to: {output_dir}")

        # 保存统计结果
        stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Total frames evaluated: {len(all_distances)}\n")
            f.write(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels\n")
            f.write(f"Min distance: {np.min(all_distances):.2f} pixels\n")
            f.write(f"Max distance: {np.max(all_distances):.2f} pixels\n")
            f.write(f"\nIndividual distances:\n")
            for i, dist in enumerate(all_distances):
                f.write(f"Frame {i+1}: {dist:.2f}\n")


def main():
    """主函数"""
    # 配置
    LORA_CHECKPOINT = "/share/project/lpy/BridgeVLA/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_lora/epoch-55.safetensors"
    DATA_ROOT = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"
    OUTPUT_DIR = "/share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/55_now"

    print("=== Heatmap Inference Test ===")

    # 创建推断引擎
    inference_engine = HeatmapInference(
        lora_checkpoint_path=LORA_CHECKPOINT,
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # 单个测试示例
    print("\n=== Single Image Test ===")
    test_image_path = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf/trail_0/bgr_images/000000.png"

    if os.path.exists(test_image_path):
        test_image = Image.open(test_image_path)
        test_prompt = "put the lion on the top shelf"

        pred_sequence = inference_engine.predict_heatmap_sequence(
            input_image=test_image,
            prompt=test_prompt,
            num_frames=5,
            seed=42
        )

        # 保存单个测试结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_image.save(os.path.join(OUTPUT_DIR, "single_test_input.png"))

        for i, frame in enumerate(pred_sequence):
            frame.save(os.path.join(OUTPUT_DIR, f"single_test_pred_frame_{i}.png"))

        # 保存详细的热力图可视化
        inference_engine.save_heatmap_visualization(
            output_path=os.path.join(OUTPUT_DIR, "single_test_detailed_analysis.png"),
            original_image=test_image,
            colormap_sequence=pred_sequence,
            colormap_name='viridis'
        )

        print(f"Single test results saved to {OUTPUT_DIR}")
    else:
        print(f"Test image not found: {test_image_path}")

    # 数据集测试
    print("\n=== Dataset Test ===")
    test_on_dataset(
        inference_engine=inference_engine,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        num_test_samples=5,  # 测试5个样本
        sequence_length=5
    )


if __name__ == "__main__":
    main()