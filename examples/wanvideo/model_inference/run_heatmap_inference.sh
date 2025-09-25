#!/bin/bash
cd /share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio/examples/wanvideo/model_inference
# 设置CoppeliaSim环境变量
export COPPELIASIM_ROOT="/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export DISPLAY=":1.0"

# 打印环境变量确认
echo "Environment variables set:"
echo "COPPELIASIM_ROOT: $COPPELIASIM_ROOT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "QT_QPA_PLATFORM_PLUGIN_PATH: $QT_QPA_PLATFORM_PLUGIN_PATH"
echo "DISPLAY: $DISPLAY"

# 检查CoppeliaSim库文件是否存在
if [ -f "$COPPELIASIM_ROOT/libcoppeliaSim.so.1" ]; then
    echo "✓ CoppeliaSim library found"
else
    echo "✗ CoppeliaSim library not found at $COPPELIASIM_ROOT/libcoppeliaSim.so.1"
    find "$COPPELIASIM_ROOT" -name "*coppelia*" -type f 2>/dev/null | head -5
fi

echo "================================"
echo "Running heatmap inference..."
echo "================================"

# 运行Python脚本
python /share/project/lpy/BridgeVLA/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference.py