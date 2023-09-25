export WDS_CACHE_SIZE=1000
export WDS_CACHE=/local/real/mandi/model_cache
export WORLD_SIZE=3
torchrun --nnodes=1 --nproc_per_node=3 real2code_train.py \
  --lm_path /home/mandi/codellama/CodeLlama-7b \
  --tokenizer_path /home/mandi/codellama/CodeLlama-7b \
  --cross_attn_every_n_layers 8 \
  --dataset_resampled \
  --batch_size_mujoco 16 \
  --train_num_samples_mujoco 418 \
  --loss_multiplier_laion 0.2 \
  --workers 4 --num_epochs 480 \
  --warmup_steps  187 \
  --fsdp --fsdp_use_orig_params \
  --run_name /local/real/mandi/flg_data/test-Codellama-no-val --report_to_wandb \
  # --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
  # --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
