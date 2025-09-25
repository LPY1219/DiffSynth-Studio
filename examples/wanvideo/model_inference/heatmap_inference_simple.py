"""
简化版热力图推断脚本
不依赖复杂的数据集，直接处理单张图像
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple
import glob

# 设置环境变量
os.environ["COPPELIASIM_ROOT"] = "/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["COPPELIASIM_ROOT"]
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
os.environ["DISPLAY"] = ":1.0"

# 添加项目路径
sys.path.append("/share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio")

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict


class SimpleHeatmapInference:
    """简化版热力图推断类"""

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

        # 加载LoRA权重
        print(f"Loading LoRA checkpoint: {lora_checkpoint_path}")
        self._load_lora_weights()

        print("Pipeline initialized successfully!")

    def _load_lora_weights(self):
        """加载LoRA权重"""
        if not os.path.exists(self.lora_checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {self.lora_checkpoint_path}")

        # 加载LoRA状态字典
        lora_state_dict = load_state_dict(self.lora_checkpoint_path)

        # 映射LoRA键名
        mapped_state_dict = {}
        for key, value in lora_state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                mapped_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                mapped_state_dict[key] = value

        # 加载到DIT模型
        load_result = self.pipe.dit.load_state_dict(mapped_state_dict, strict=False)
        print(f"LoRA checkpoint loaded: {len(mapped_state_dict)} keys")
        if len(load_result[1]) > 0:
            print(f"Warning: Unexpected keys in LoRA checkpoint: {load_result[1]}")

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

    def find_peak_position(self, heatmap_image: Image.Image) -> Tuple[int, int]:
        """
        找到热力图中的峰值位置

        Args:
            heatmap_image: 热力图PIL图像

        Returns:
            (x, y) 峰值位置坐标
        """
        # 转换为numpy数组并转为灰度
        heatmap_array = np.array(heatmap_image.convert('L'))

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

    def visualize_prediction(self,
                           input_image: Image.Image,
                           predicted_frames: List[Image.Image],
                           prompt: str,
                           output_path: str = None):
        """
        可视化预测结果

        Args:
            input_image: 输入图像
            predicted_frames: 预测的帧序列
            prompt: 提示文本
            output_path: 输出路径
        """
        num_frames = len(predicted_frames)
        fig, axes = plt.subplots(2, max(num_frames, 1), figsize=(4*num_frames, 8))

        if num_frames == 1:
            axes = axes.reshape(-1, 1)

        # 第一行：输入图像和预测序列
        for i in range(num_frames):
            if i == 0:
                # 显示输入图像
                axes[0, i].imshow(input_image)
                axes[0, i].set_title('Input Image')
                axes[0, i].axis('off')
            else:
                axes[0, i].axis('off')

        # 第二行：预测的热力图序列
        for i, pred_frame in enumerate(predicted_frames):
            axes[1, i].imshow(pred_frame)

            # 找到并标记峰值
            peak_pos = self.find_peak_position(pred_frame)
            axes[1, i].plot(peak_pos[0], peak_pos[1], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)

            axes[1, i].set_title(f'Predicted Frame {i+1}\nPeak: ({peak_pos[0]}, {peak_pos[1]})')
            axes[1, i].axis('off')

        plt.suptitle(f'Heatmap Prediction\nPrompt: {prompt}', fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()


def test_simple_inference():
    """简单的推断测试"""
    # 配置
    LORA_CHECKPOINT = "/share/project/lpy/BridgeVLA/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_lora/epoch-100.safetensors"
    OUTPUT_DIR = "./simple_heatmap_results"

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Simple Heatmap Inference Test ===")

    # 创建推断引擎
    inference_engine = SimpleHeatmapInference(
        lora_checkpoint_path=LORA_CHECKPOINT,
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # 测试图像路径（根据您的数据结构调整）
    test_image_paths = glob.glob("/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf/episode_*/images/wrist_image_left.png")

    if not test_image_paths:
        # 如果找不到测试图像，创建一个示例图像
        print("No test images found, creating a dummy image...")
        test_image = Image.new('RGB', (256, 256), color='gray')
        test_prompt = "Put the lion on the top shelf"

        pred_sequence = inference_engine.predict_heatmap_sequence(
            input_image=test_image,
            prompt=test_prompt,
            num_frames=5,
            seed=42
        )

        # 可视化结果
        inference_engine.visualize_prediction(
            input_image=test_image,
            predicted_frames=pred_sequence,
            prompt=test_prompt,
            output_path=os.path.join(OUTPUT_DIR, "dummy_test_result.png")
        )
    else:
        # 使用真实测试图像
        for i, image_path in enumerate(test_image_paths[:3]):  # 测试前3张图像
            print(f"\nTesting image {i+1}: {image_path}")

            try:
                test_image = Image.open(image_path)
                test_prompt = "Put the lion on the top shelf"  # 根据您的数据调整

                pred_sequence = inference_engine.predict_heatmap_sequence(
                    input_image=test_image,
                    prompt=test_prompt,
                    num_frames=5,
                    seed=42
                )

                # 可视化结果
                output_path = os.path.join(OUTPUT_DIR, f"test_result_{i+1}.png")
                inference_engine.visualize_prediction(
                    input_image=test_image,
                    predicted_frames=pred_sequence,
                    prompt=test_prompt,
                    output_path=output_path
                )

                # 保存单独的帧
                test_image.save(os.path.join(OUTPUT_DIR, f"test_{i+1}_input.png"))
                for j, frame in enumerate(pred_sequence):
                    frame.save(os.path.join(OUTPUT_DIR, f"test_{i+1}_pred_frame_{j+1}.png"))

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    test_simple_inference()