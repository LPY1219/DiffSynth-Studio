# Heatmap Inference Scripts

这里提供了两个用于热力图序列预测的推断脚本：

## 文件说明

### 1. `heatmap_inference.py` (完整版)
- **功能**: 完整的推断和评估脚本
- **特点**:
  - 支持单张图像推断
  - 支持数据集批量测试
  - 自动计算预测峰值与真实峰值的距离
  - 生成详细的可视化比较结果
  - 输出评估统计信息

### 2. `heatmap_inference_simple.py` (简化版)
- **功能**: 简化的推断脚本，不依赖复杂数据集
- **特点**:
  - 直接处理单张图像
  - 基本的可视化功能
  - 更容易调试和使用

## 使用方法

### 环境准备
确保已安装所需依赖：
```bash
pip install torch torchvision matplotlib pillow numpy opencv-python
```

### 简化版使用（推荐开始使用）

```bash
cd /share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio/examples/wanvideo/model_inference
python heatmap_inference_simple.py
```

### 完整版使用

```bash
cd /share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio/examples/wanvideo/model_inference
python heatmap_inference.py
```

## 核心功能

### 1. HeatmapInference 类

#### 初始化
```python
inference_engine = HeatmapInference(
    lora_checkpoint_path="/path/to/your/lora/checkpoint.safetensors",
    model_base_path="/share/project/lpy/huggingface/Wan_2_2_TI2V_5B",
    device="cuda",
    torch_dtype=torch.bfloat16
)
```

#### 单张图像推断
```python
from PIL import Image

# 加载输入图像
input_image = Image.open("your_input_image.png")
prompt = "Put the lion on the top shelf"

# 生成热力图序列
predicted_frames = inference_engine.predict_heatmap_sequence(
    input_image=input_image,
    prompt=prompt,
    num_frames=5,
    height=256,
    width=256,
    seed=42
)
```

#### 峰值位置检测
```python
# 找到热力图中的峰值位置
peak_x, peak_y = inference_engine.find_peak_position(heatmap_image)

# 计算两个峰值之间的距离
distance = inference_engine.calculate_peak_distance(pred_peak, gt_peak)
```

### 2. 数据集测试功能

```python
# 在数据集上进行批量测试
test_on_dataset(
    inference_engine=inference_engine,
    data_root="/path/to/your/dataset",
    output_dir="./inference_results",
    num_test_samples=10,
    sequence_length=5
)
```

## 输出结果

### 可视化输出
- **输入图像**: 原始的RGB图像
- **预测热力图序列**: 生成的热力图帧序列
- **峰值标记**: 红色星号标记峰值位置
- **叠加比较**: 真实值与预测值的叠加显示

### 评估指标
- **峰值距离**: 预测峰值与真实峰值之间的欧几里得距离（像素）
- **平均距离**: 所有测试帧的平均峰值距离
- **标准差**: 距离的标准差
- **最小/最大距离**: 距离的范围

### 输出文件
```
inference_results/
├── sample_001_comparison.png      # 样本比较可视化
├── sample_001_input.png          # 输入图像
├── sample_002_comparison.png
├── ...
└── evaluation_stats.txt          # 评估统计结果
```

## 配置说明

### 重要参数
- `lora_checkpoint_path`: 训练好的LoRA模型路径
- `model_base_path`: 基础Wan2.2模型路径
- `num_frames`: 预测的热力图帧数
- `height/width`: 输出图像尺寸
- `seed`: 随机种子（用于可重现结果）

### 数据路径
根据您的实际情况修改以下路径：
- LoRA检查点: `/share/project/lpy/BridgeVLA/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_lora/epoch-100.safetensors`
- 测试数据: `/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf`

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch size或使用CPU
2. **模型文件未找到**: 检查路径是否正确
3. **数据集导入失败**: 使用简化版脚本

### 调试技巧
1. 首先使用简化版脚本测试基本功能
2. 检查LoRA权重是否正确加载
3. 验证输入图像格式和尺寸
4. 确认prompt格式与训练时一致

## 扩展使用

### 自定义推断
```python
# 创建推断引擎
inference_engine = SimpleHeatmapInference(lora_checkpoint_path="your_checkpoint.safetensors")

# 加载您的图像
input_image = Image.open("your_image.png")

# 生成预测
predictions = inference_engine.predict_heatmap_sequence(
    input_image=input_image,
    prompt="your custom prompt",
    num_frames=5,
    seed=42
)

# 可视化结果
inference_engine.visualize_prediction(
    input_image=input_image,
    predicted_frames=predictions,
    prompt="your custom prompt",
    output_path="result.png"
)
```