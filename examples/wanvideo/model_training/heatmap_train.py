"""
Heatmap Training Script for Wan2.2
基于原始train.py，专门用于热力图序列生成训练
"""

import torch
import os
import json
import sys

# 设置CoppeliaSim环境变量（必须在其他导入之前）
os.environ["COPPELIASIM_ROOT"] = "/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["COPPELIASIM_ROOT"]
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
os.environ["DISPLAY"] = ":1.0"

# SwanLab导入（可选）
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Install with: pip install swanlab")

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import torch.utils.data

# 添加trainers路径以导入我们的自定义数据集
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HeatmapWanTrainingModule(DiffusionTrainingModule):
    """
    热力图专用的Wan训练模块
    继承原始WanTrainingModule，针对热力图序列生成进行优化
    """

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        """
        预处理输入数据，专门针对热力图数据格式
        """
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # 热力图序列作为视频输入
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # 训练相关参数
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs - 对于热力图任务，主要是input_image
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # 使用首帧RGB图像作为条件输入
                inputs_shared["input_image"] = data["input_image"]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        """
        前向传播，计算损失
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    def log_to_swanlab(self, loss_value, step, swanlab_run=None):
        """
        记录损失到SwanLab
        """
        if swanlab_run is not None and SWANLAB_AVAILABLE:
            try:
                import swanlab
                # 只记录有效的数值，避免None值导致的错误
                log_data = {
                    "train_loss": loss_value,
                    "step": step
                }

                # 如果有学习率方法且返回有效值，才记录学习率
                if hasattr(self, 'get_current_lr'):
                    lr = self.get_current_lr()
                    if lr is not None:
                        log_data["learning_rate"] = lr

                swanlab.log(log_data, step=step)
                print(f"SwanLab logged: step={step}, loss={loss_value:.4f}")
            except Exception as e:
                print(f"Warning: Failed to log to SwanLab: {e}")
                import traceback
                traceback.print_exc()


def launch_optimized_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    args=None,
):
    """
    优化版本的训练任务启动器，专门针对40GB A100进行内存和性能优化
    """
    if args is None:
        raise ValueError("args is required for optimized training")

    # 参数提取
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    train_batch_size = args.train_batch_size

    # 只在主进程打印配置信息（避免多进程重复打印）
    temp_accelerator_for_check = Accelerator()
    if temp_accelerator_for_check.is_main_process:
        print(f"Optimized training configuration:")
        print(f"  - Batch size: {train_batch_size}")
        print(f"  - Data workers: {num_workers}")
        print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  - Gradient checkpointing offload: {args.use_gradient_checkpointing_offload}")

    # 优化的数据加载器配置（单进程模式）
    dataloader_kwargs = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': num_workers == 0,  # 单进程时启用pin memory
        'persistent_workers': num_workers > 0,  # 只在多进程时启用
        'prefetch_factor': 2 if num_workers > 0 else None,  # 只在多进程时预取
        'drop_last': True,  # 丢弃最后不完整的batch
        'collate_fn': lambda x: x[0],
    }

    if temp_accelerator_for_check.is_main_process:
        print(f"DataLoader configuration: {dataloader_kwargs}")

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,  # 稍微提高数值稳定性
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Accelerator配置
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    # 准备训练组件
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # 显存优化：启用自动清理无用变量
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # 训练循环
    for epoch_id in range(num_epochs):
        model.train()
        epoch_loss = 0
        step_count = 0

        # 添加进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")

        for step, data in enumerate(pbar):
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                # 前向传播
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)

                # 反向传播
                accelerator.backward(loss)

                # 梯度裁剪
                if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 优化器步骤
                optimizer.step()
                scheduler.step()

                # 记录损失
                epoch_loss += loss.item()
                step_count += 1

                # SwanLab日志记录 - 只在主进程且满足logging_steps频率时记录
                global_step = step + epoch_id * len(dataloader)
                should_log = (accelerator.is_main_process and
                             hasattr(args, 'swanlab_run') and args.swanlab_run is not None and
                             hasattr(args, 'logging_steps') and global_step % args.logging_steps == 0)

                if should_log:
                    print(f"Logging to SwanLab at step {global_step}")
                    # 记录训练指标到SwanLab
                    try:
                        import swanlab
                        # 获取当前学习率
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

                        log_data = {
                            "train_loss": loss.item(),
                            "learning_rate": current_lr,
                            "epoch": epoch_id,
                            "step": global_step
                        }
                        swanlab.log(log_data, step=global_step)
                        print(f"SwanLab logged: step={global_step}, loss={loss.item():.4f}, lr={current_lr:.2e}")
                    except Exception as e:
                        print(f"Warning: Failed to log to SwanLab: {e}")
                elif global_step % args.logging_steps == 0:
                    print(f"Step {global_step}: main_process={accelerator.is_main_process}, swanlab_run={args.swanlab_run is not None}")

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_loss/step_count:.4f}"
                })

                # 定期保存检查点（只在save_steps > 0时保存）
                if save_steps > 0:
                    model_logger.on_step_end(accelerator, model, save_steps)

                # 定期清理显存
                if step % 10 == 0:
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

        # 每个epoch结束时的处理 - 总是保存epoch检查点
        model_logger.on_epoch_end(accelerator, model, epoch_id)

        accelerator.print(f"Epoch {epoch_id+1} completed. Average loss: {epoch_loss/step_count:.4f}")

    # 训练结束处理（只在save_steps > 0时保存最后的step检查点）
    if save_steps > 0:
        model_logger.on_training_end(accelerator, model, save_steps)

    # 最终清理显存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


def create_heatmap_parser():
    """
    创建热力图训练专用的参数解析器
    """
    import argparse
    parser = argparse.ArgumentParser(description="Heatmap sequence training for Wan2.2")

    # 从wan_parser复制必要的参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="", help="Remove prefix in checkpoint.")
    parser.add_argument("--trainable_models", type=str, default="", help="Trainable models.")
    parser.add_argument("--lora_base_model", type=str, default="", help="LoRA base model.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="LoRA target modules.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_checkpoint", type=str, default="", help="LoRA checkpoint.")
    parser.add_argument("--extra_inputs", type=str, default="", help="Extra inputs.")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", help="Use gradient checkpointing offload.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--find_unused_parameters", action="store_true", help="Find unused parameters.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Dataset num workers.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--model_paths", type=str, default="", help="Model paths.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="", help="Model ID with origin paths.")
    parser.add_argument("--height", type=int, default=256, help="Image height.")
    parser.add_argument("--width", type=int, default=256, help="Image width.")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Dataset repeat.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")

    # 添加热力图专用参数
    parser.add_argument("--heatmap_data_root", type=str, required=True,
                       help="Root directory containing robot trajectory data")
    parser.add_argument("--sequence_length", type=int, default=10,
                       help="Length of heatmap sequence to predict")
    parser.add_argument("--step_interval", type=int, default=1,
                       help="Interval between trajectory steps")
    parser.add_argument("--min_trail_length", type=int, default=15,
                       help="Minimum trajectory length requirement")
    parser.add_argument("--heatmap_sigma", type=float, default=1.5,
                       help="Standard deviation for Gaussian heatmap generation")
    parser.add_argument("--colormap_name", type=str, default="viridis",
                       help="Colormap name for heatmap conversion")
    parser.add_argument("--scene_bounds", type=str,
                       default="0,-0.45,-0.05,0.8,0.55,0.6",
                       help="Scene bounds as comma-separated values")
    parser.add_argument("--transform_augmentation_xyz", type=str,
                       default="0.1,0.1,0.1",
                       help="XYZ augmentation range as comma-separated values")
    parser.add_argument("--transform_augmentation_rpy", type=str,
                       default="0.0,0.0,20.0",
                       help="RPY augmentation range as comma-separated values")
    parser.add_argument("--disable_augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode (use fewer data)")

    # SwanLab相关参数
    parser.add_argument("--enable_swanlab", action="store_true",
                       help="Enable SwanLab logging")
    parser.add_argument("--swanlab_api_key", type=str, default="h1x6LOLp5qGLTfsPuB7Qw",
                       help="SwanLab API key")
    parser.add_argument("--swanlab_project", type=str, default="wan2.2-heatmap-training",
                       help="SwanLab project name")
    parser.add_argument("--swanlab_experiment", type=str, default="heatmap-lora",
                       help="SwanLab experiment name")

    return parser


def parse_float_list(s: str, name: str):
    """
    解析逗号分隔的浮点数列表
    """
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid format for {name}: {s}. Expected comma-separated numbers.") from e


if __name__ == "__main__":
    parser = create_heatmap_parser()
    args = parser.parse_args()

    print("="*60)
    print("HEATMAP SEQUENCE TRAINING FOR WAN2.2")
    print("="*60)
    print(f"Data root: {args.heatmap_data_root}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Colormap: {args.colormap_name}")
    print(f"Output path: {args.output_path}")
    print(f"Debug mode: {args.debug_mode}")
    print(f"SwanLab enabled: {args.enable_swanlab and not args.debug_mode}")
    print("="*60)

    # 初始化SwanLab（只在主进程进行）
    swanlab_run = None

    # 创建一个临时的accelerator来检查是否为主进程
    temp_accelerator = Accelerator()
    is_main_process = temp_accelerator.is_main_process

    if is_main_process and args.enable_swanlab and not args.debug_mode and SWANLAB_AVAILABLE:
        try:
            print("Initializing SwanLab...")
            print(f"API Key: {args.swanlab_api_key[:8]}***")
            print(f"Project: {args.swanlab_project}")
            print(f"Experiment: {args.swanlab_experiment}")

            swanlab.login(api_key=args.swanlab_api_key)
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config={
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_epochs,
                    "sequence_length": args.sequence_length,
                    "train_batch_size": args.train_batch_size,
                    "lora_rank": args.lora_rank,
                    "heatmap_sigma": args.heatmap_sigma,
                    "colormap_name": args.colormap_name,
                    "height": args.height,
                    "width": args.width,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "logging_steps": args.logging_steps,
                }
            )
            print(f"✅ SwanLab initialized successfully!")
            print(f"   Project: {args.swanlab_project}")
            print(f"   Experiment: {args.swanlab_experiment}")
            print(f"   Logging frequency: every {args.logging_steps} steps")

            # 测试一次日志记录
            swanlab.log({"test": 1.0}, step=0)
            print("✅ Test log sent to SwanLab")

        except Exception as e:
            print(f"❌ Failed to initialize SwanLab: {e}")
            import traceback
            traceback.print_exc()
            swanlab_run = None
    elif is_main_process and args.enable_swanlab and args.debug_mode:
        print("SwanLab disabled in debug mode")
    elif is_main_process and args.enable_swanlab and not SWANLAB_AVAILABLE:
        print("Warning: SwanLab requested but not available. Install with: pip install swanlab")

    # 解析参数
    scene_bounds = parse_float_list(args.scene_bounds, "scene_bounds")
    transform_augmentation_xyz = parse_float_list(args.transform_augmentation_xyz, "transform_augmentation_xyz")
    transform_augmentation_rpy = parse_float_list(args.transform_augmentation_rpy, "transform_augmentation_rpy")

    # 创建热力图数据集
    if is_main_process:
        print("Creating heatmap dataset...")
    try:
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
            data_root=args.heatmap_data_root,
            sequence_length=args.sequence_length,
            step_interval=args.step_interval,
            min_trail_length=args.min_trail_length,
            image_size=(args.height, args.width),
            sigma=args.heatmap_sigma,
            augmentation=not args.disable_augmentation,
            mode="train",
            scene_bounds=scene_bounds,
            transform_augmentation_xyz=transform_augmentation_xyz,
            transform_augmentation_rpy=transform_augmentation_rpy,
            debug=args.debug_mode,
            colormap_name=args.colormap_name,
            repeat=args.dataset_repeat
        )
        if is_main_process:
            print(f"Dataset created successfully with {len(dataset)} samples")

            # 测试数据加载
            print("Testing data loading...")
            test_sample = dataset[0]
            print(f"Sample keys: {list(test_sample.keys())}")
            print(f"Video frames: {len(test_sample['video'])}")
            print(f"First frame size: {test_sample['video'][0].size}")
            print(f"Prompt: {test_sample['prompt'][:50]}...")

    except Exception as e:
        if is_main_process:
            print(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
        exit(1)

    # 创建训练模块
    if is_main_process:
        print("Creating training module...")
    try:
        # 处理空字符串参数，避免导致错误
        model_id_with_origin_paths = args.model_id_with_origin_paths if args.model_id_with_origin_paths else None
        lora_checkpoint = args.lora_checkpoint if args.lora_checkpoint else None
        trainable_models = args.trainable_models if args.trainable_models else None

        model = HeatmapWanTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            trainable_models=trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=lora_checkpoint,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
        if is_main_process:
            print("Training module created successfully")

    except Exception as e:
        if is_main_process:
            print(f"Error creating training module: {e}")
            import traceback
            traceback.print_exc()
        exit(1)

    # 创建模型日志器
    if is_main_process:
        print("Setting up model logger...")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    # 启动训练
    if is_main_process:
        print("Starting training...")
        print(f"Optimizing data loading with {args.dataset_num_workers} workers...")
    try:
        # 如果SwanLab可用，添加到args中
        if swanlab_run is not None:
            args.swanlab_run = swanlab_run
        else:
            args.swanlab_run = None

        # 使用优化版本的训练函数，专门针对40GB A100优化
        launch_optimized_training_task(dataset, model, model_logger, args=args)

        if is_main_process:
            print("Training completed successfully!")

        # 结束SwanLab实验
        if swanlab_run is not None:
            swanlab_run.finish()
            if is_main_process:
                print("SwanLab experiment finished")

    except Exception as e:
        if is_main_process:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        # 确保SwanLab实验正确结束
        if swanlab_run is not None:
            swanlab_run.finish()

        exit(1)