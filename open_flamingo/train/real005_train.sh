# NOTE: on real005, IPV4 issue seems to go away with: 
export NCCL_SOCKET_IFNAME=enp37s0f0; export MASTER_ADR=172.24.72.83  
export WDS_CACHE_SIZE=800
export WDS_CACHE=/local/real/mandi/model_cache
export WORLD_SIZE=3
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE}  real2code_train.py \
  --lm_path /home/mandi/codellama/CodeLlama-7b \
  --tokenizer_path /home/mandi/codellama/CodeLlama-7b \
  --dataset_resampled \
  --batch_size_mujoco 1 \
  --train_num_samples_mujoco 424 --workers 1 --num_epochs 30 \
  --warmup_steps  187 \
  --fsdp --fsdp_use_orig_params \
  --cross_attn_every_n_layers 16 --gradient_accumulation_steps 4 \
  --run_name /local/real/mandi/flg_data/Codellama-16Layer-1Image --learning_rate 3e-4 --report_to_wandb
  # --cross_attn_every_n_layers 16 \
  # --run_name /local/real/mandi/flg_data/test-Codellama-16Layer
  # --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
  # --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
