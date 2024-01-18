
export WDS_CACHE_SIZE=800
export WDS_CACHE=/local/real/mandi/model_cache
export WORLD_SIZE=3
export CUDA_VISIBLE_DEVICES=1,2,3


## using pore
# torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE}  real2code_train.py \
#   --lm_path /local/real/mandi/codellama_data/CodeLlama-7b \
#   --tokenizer_path /local/real/mandi/codellama_data/CodeLlama-7b \
#   --dataset_resampled \
#   --batch_size_mujoco 1 \
#   --train_num_samples_mujoco 424 --workers 1 --num_epochs 30 \
#   --warmup_steps  187 \
#   --fsdp --fsdp_use_orig_params \
#   --cross_attn_every_n_layers -1 --gradient_accumulation_steps 32 \
#   --run_name /local/real/mandi/flamingo_data/rand_bbox --learning_rate 1e-4 --report_to_wandb
#   # --cross_attn_every_n_layers 16 \
