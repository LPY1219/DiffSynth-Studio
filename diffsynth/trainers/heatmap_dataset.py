"""
Heatmap Dataset for Wan2.2 Training
适配RobotTrajectoryDataset到UnifiedDataset格式的热力图数据集
"""

import sys
import os
import torch
from typing import Dict, Any, Optional
from .unified_dataset import UnifiedDataset
from .heatmap_utils import prepare_heatmap_data_for_wan

# 设置CoppeliaSim环境变量
os.environ["COPPELIASIM_ROOT"] = "/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["COPPELIASIM_ROOT"]
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
os.environ["DISPLAY"] = ":1.0"

# 添加single_view项目路径
sys.path.append("/share/project/lpy/BridgeVLA/Wan/single_view")

try:
    from data.dataset import RobotTrajectoryDataset, ProjectionInterface
except ImportError as e:
    print(f"Warning: Could not import RobotTrajectoryDataset: {e}")
    print("Please ensure the single_view project is properly set up.")
    RobotTrajectoryDataset = None
    ProjectionInterface = None


class HeatmapUnifiedDataset(UnifiedDataset):
    """
    热力图专用的UnifiedDataset实现
    包装RobotTrajectoryDataset并转换为Wan2.2训练所需格式
    """

    def __init__(self,
                 robot_dataset_config: Dict[str, Any],
                 colormap_name: str = 'viridis',
                 repeat: int = 1,
                 **kwargs):
        """
        初始化热力图数据集

        Args:
            robot_dataset_config: RobotTrajectoryDataset的配置参数
            colormap_name: 使用的colormap名称
            repeat: 数据重复次数
            **kwargs: 其他UnifiedDataset参数
        """
        self.colormap_name = colormap_name
        self.robot_dataset_config = robot_dataset_config

        # 检查依赖
        if RobotTrajectoryDataset is None:
            raise ImportError("RobotTrajectoryDataset not available. Please check single_view project setup.")

        # 创建机器人轨迹数据集
        self.robot_dataset = self._create_robot_dataset()

        # 初始化父类（使用虚拟参数，因为我们会重写关键方法）
        super().__init__(
            base_path="",  # 使用空字符串而不是None
            metadata_path=None,
            repeat=repeat,
            data_file_keys=(),
            main_data_operator=lambda x: x,
            **kwargs
        )

        # 重置数据
        self.data = []
        self.cached_data = []
        self.load_from_cache = False

        print(f"HeatmapUnifiedDataset initialized with {len(self.robot_dataset)} samples")

    def load_metadata(self, metadata_path):
        """
        重写父类的load_metadata方法，跳过文件搜索
        """
        # 我们不需要加载metadata，因为我们使用RobotTrajectoryDataset
        pass

    def _create_robot_dataset(self) -> RobotTrajectoryDataset:
        """
        创建RobotTrajectoryDataset实例
        """
        # 确保必要的参数存在
        required_params = ['data_root']
        for param in required_params:
            if param not in self.robot_dataset_config:
                raise ValueError(f"Missing required parameter: {param}")

        # 创建投影接口
        if 'projection_interface' not in self.robot_dataset_config:
            self.robot_dataset_config['projection_interface'] = ProjectionInterface()

        return RobotTrajectoryDataset(**self.robot_dataset_config)

    def __len__(self) -> int:
        """
        返回数据集长度
        """
        return len(self.robot_dataset) * self.repeat

    def __getitem__(self, data_id: int) -> Dict[str, Any]:
        """
        获取训练样本，转换为Wan2.2训练格式

        Args:
            data_id: 样本索引

        Returns:
            转换后的数据字典，包含 'prompt', 'video', 'input_image'
        """
        # 获取原始样本
        robot_sample = self.robot_dataset[data_id % len(self.robot_dataset)]

        # 提取数据
        rgb_image = robot_sample['rgb_image']  # (3, H, W)
        heatmap_sequence = robot_sample['heatmap_sequence']  # (T, H, W)
        instruction = robot_sample['instruction']

        # 转换为Wan2.2格式
        wan_data = prepare_heatmap_data_for_wan(
            rgb_image=rgb_image,
            heatmap_sequence=heatmap_sequence,
            instruction=instruction,
            colormap_name=self.colormap_name
        )

        return wan_data

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        获取样本的详细信息
        """
        robot_sample = self.robot_dataset[idx % len(self.robot_dataset)]
        return {
            'robot_sample_info': robot_sample,
            'colormap_name': self.colormap_name,
            'dataset_type': 'heatmap'
        }


class HeatmapDatasetFactory:
    """
    热力图数据集工厂类，用于创建不同配置的数据集
    """

    @staticmethod
    def create_robot_trajectory_dataset(
        data_root: str,
        sequence_length: int = 10,
        step_interval: int = 1,
        min_trail_length: int = 15,
        image_size: tuple = (256, 256),
        sigma: float = 1.5,
        augmentation: bool = True,
        mode: str = "train",
        scene_bounds: list = [0, -0.45, -0.05, 0.8, 0.55, 0.6],
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        debug: bool = False,
        colormap_name: str = 'viridis',
        repeat: int = 1,
        **kwargs
    ) -> HeatmapUnifiedDataset:
        """
        创建机器人轨迹热力图数据集

        Args:
            data_root: 数据根目录
            sequence_length: 热力图序列长度
            step_interval: step采样间隔
            min_trail_length: 最小轨迹长度
            image_size: 图像尺寸
            sigma: 热力图高斯标准差
            augmentation: 是否使用数据增强
            mode: 训练模式
            scene_bounds: 场景边界
            transform_augmentation_xyz: xyz变换增强范围
            transform_augmentation_rpy: rpy变换增强范围
            debug: 调试模式
            colormap_name: colormap名称
            repeat: 数据重复次数
            **kwargs: 其他参数

        Returns:
            HeatmapUnifiedDataset实例
        """
        # 检查数据目录
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")

        # 创建投影接口
        projection_interface = ProjectionInterface(
            img_size=image_size[0],
            rend_three_views=True,
            add_depth=False
        )

        # 机器人数据集配置
        robot_config = {
            'data_root': data_root,
            'projection_interface': projection_interface,
            'sequence_length': sequence_length,
            'step_interval': step_interval,
            'min_trail_length': min_trail_length,
            'image_size': image_size,
            'sigma': sigma,
            'augmentation': augmentation,
            'mode': mode,
            'scene_bounds': scene_bounds,
            'transform_augmentation_xyz': transform_augmentation_xyz,
            'transform_augmentation_rpy': transform_augmentation_rpy,
            'debug': debug
        }

        return HeatmapUnifiedDataset(
            robot_dataset_config=robot_config,
            colormap_name=colormap_name,
            repeat=repeat,
            **kwargs
        )

    @staticmethod
    def create_from_config_file(config_path: str) -> HeatmapUnifiedDataset:
        """
        从配置文件创建数据集
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        return HeatmapDatasetFactory.create_robot_trajectory_dataset(**config)


# 测试代码
if __name__ == "__main__":
    # 测试数据集创建
    try:
        print("Testing HeatmapUnifiedDataset creation...")

        # 使用示例数据路径（需要根据实际情况调整）
        test_data_root = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"

        if os.path.exists(test_data_root):
            dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
                data_root=test_data_root,
                sequence_length=5,
                debug=True,  # 调试模式，只使用少量数据
                repeat=1
            )

            print(f"Dataset created successfully with {len(dataset)} samples")

            # 测试获取样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Prompt: {sample['prompt']}")
                print(f"Video frames: {len(sample['video'])}")
                print(f"Input image type: {type(sample['input_image'])}")

        else:
            print(f"Test data directory not found: {test_data_root}")
            print("Skipping test...")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("Test completed.")