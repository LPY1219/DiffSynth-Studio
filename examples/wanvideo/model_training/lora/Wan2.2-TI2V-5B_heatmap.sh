#!/bin/bash

# Heatmap Sequence Training Script for Wan2.2-TI2V-5B
# 基于原始Wan2.2-TI2V-5B.sh，专门用于热力图序列生成训练

# 设置conda环境
source /share/project/lpy/miniconda3/etc/profile.d/conda.sh
conda activate test

# 设置CoppeliaSim环境变量
export COPPELIASIM_ROOT=/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

# 内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:16
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4  # 限制OpenMP线程数，避免CPU过载影响GPU利用率

# 多进程启动方法设置（解决CUDA多进程问题）
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=/share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio:$PYTHONPATH

# 设置工作目录
cd /share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio

# ==============================================
# GPU配置 - 支持单机多卡训练
# ==============================================
# 8张A100训练配置（如需使用更少GPU，请修改以下配置）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# 其他常用配置示例：
# export CUDA_VISIBLE_DEVICES=0; NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1; NUM_GPUS=2
# 四GPU: export CUDA_VISIBLE_DEVICES=0,1,2,3; NUM_GPUS=4

# 数据路径配置
HEATMAP_DATA_ROOT="/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"
OUTPUT_PATH="/share/project/lpy/BridgeVLA/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_lora"

# 模型路径配置
MODEL_BASE_PATH="/share/project/lpy/huggingface/Wan_2_2_TI2V_5B"

# 热力图专用参数
SEQUENCE_LENGTH=5          # 热力图序列长度
STEP_INTERVAL=1           # 轨迹步长间隔
MIN_TRAIL_LENGTH=10       # 最小轨迹长度
HEATMAP_SIGMA=1.5         # 高斯热力图标准差
COLORMAP_NAME="viridis"   # colormap名称

# 图像和训练参数
HEIGHT=256
WIDTH=256
NUM_FRAMES=${SEQUENCE_LENGTH}
# 优化方案：减少DATASET_REPEAT，增加EPOCHS获得更精细的控制
DATASET_REPEAT=1                     # 不重复数据集
LEARNING_RATE=1e-4
NUM_EPOCHS=200                        # 增加epochs数量（1×50 = 原来的10×5）

# 多GPU训练参数调整 - 40GB A100优化
TRAIN_BATCH_SIZE_PER_GPU=2             # 每张GPU的批次大小（调整为2）
GRADIENT_ACCUMULATION_STEPS=1           # 梯度累积步数
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))

# 内存和性能优化参数（放宽显存优化）
DATASET_NUM_WORKERS=0                   # 数据加载线程数（设为0避免CUDA多进程问题）
USE_GRADIENT_CHECKPOINTING=false        # 关闭梯度检查点（提升速度，稍微增加显存）
USE_GRADIENT_CHECKPOINTING_OFFLOAD=false # 关闭梯度检查点卸载
MIXED_PRECISION="bf16"                  # 使用bf16混合精度（保持节省显存）
DATALOADER_PIN_MEMORY=true              # 启用pin memory（加速数据传输）
PREFETCH_FACTOR=2                       # 数据预取因子（提升数据加载效率）

# LoRA参数
LORA_RANK=32
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# 数据增强参数
SCENE_BOUNDS="0,-0.45,-0.05,0.8,0.55,0.6"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"
TRANSFORM_AUG_RPY="0.0,0.0,20.0"

# SwanLab配置参数
ENABLE_SWANLAB=true                         # 是否启用SwanLab记录
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"    # SwanLab API密钥
SWANLAB_PROJECT="wan2.2-heatmap-training-single-view"   # SwanLab项目名称
SWANLAB_EXPERIMENT="heatmap-lora-$(date +%Y%m%d-%H%M%S)"  # SwanLab实验名称（添加时间戳）
DEBUG_MODE=false                            # 调试模式（为true时禁用SwanLab）

echo "================================================================"
echo "HEATMAP SEQUENCE TRAINING FOR WAN2.2-TI2V-5B"
echo "================================================================"
echo "Data root: ${HEATMAP_DATA_ROOT}"
echo "Output path: ${OUTPUT_PATH}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Image size: ${HEIGHT}x${WIDTH}"
echo "Learning rate: ${LEARNING_RATE}"
echo "LoRA rank: ${LORA_RANK}"
echo "Multi-GPU setup: ${NUM_GPUS} GPUs"
echo "Batch size per GPU: ${TRAIN_BATCH_SIZE_PER_GPU}"
echo "Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "Data workers: ${DATASET_NUM_WORKERS}"
echo "Mixed precision: ${MIXED_PRECISION}"
echo "Gradient checkpointing: ${USE_GRADIENT_CHECKPOINTING}"
echo "SwanLab enabled: ${ENABLE_SWANLAB}"
echo "Debug mode: ${DEBUG_MODE}"
echo "================================================================"

# 检查数据目录
if [ ! -d "${HEATMAP_DATA_ROOT}" ]; then
    echo "Error: Data directory not found: ${HEATMAP_DATA_ROOT}"
    echo "Please check the data path and try again."
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"

# 启动训练 - 多GPU分布式训练
accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --mixed_precision=${MIXED_PRECISION} \
  --main_process_port=29500 \
  examples/wanvideo/model_training/heatmap_train.py \
  --heatmap_data_root "${HEATMAP_DATA_ROOT}" \
  --sequence_length ${SEQUENCE_LENGTH} \
  --step_interval ${STEP_INTERVAL} \
  --min_trail_length ${MIN_TRAIL_LENGTH} \
  --heatmap_sigma ${HEATMAP_SIGMA} \
  --colormap_name "${COLORMAP_NAME}" \
  --scene_bounds "${SCENE_BOUNDS}" \
  --transform_augmentation_xyz "${TRANSFORM_AUG_XYZ}" \
  --transform_augmentation_rpy "${TRANSFORM_AUG_RPY}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --num_frames ${NUM_FRAMES} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_paths '[
    [
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
    "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
    ]' \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "input_image" \
  --train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  $(if [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "true" ]; then echo "--use_gradient_checkpointing_offload"; fi) \
  --save_steps 0 \
  --logging_steps 10 \
  --max_grad_norm 1.0 \
  --warmup_steps 100 \
  $(if [ "${DEBUG_MODE}" = "true" ]; then echo "--debug_mode"; fi) \
  $(if [ "${ENABLE_SWANLAB}" = "true" ]; then echo "--enable_swanlab"; fi) \
  --swanlab_api_key "${SWANLAB_API_KEY}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment "${SWANLAB_EXPERIMENT}"

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"