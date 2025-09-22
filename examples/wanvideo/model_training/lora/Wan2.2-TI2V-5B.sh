accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 256 \
  --width 256 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths '[
    [
        "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B/diffusion_pytorch_model-00003-of-00003.safetensors",
    ],
    "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B/models_t5_umt5-xxl-enc-bf16.pth",
    "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B/Wan2.2_VAE.pth"
    ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/share/project/lpy/BridgeVLA/logs/Wan/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"