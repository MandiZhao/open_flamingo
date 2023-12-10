export CUDA_VISIBLE_DEVICES=0,1,2
export WORLD_SIZE=3
export WDS_CACHE=/local/real/mandi/model_cache
torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} real2code_eval.py --checkpoint_path /local/real/mandi/flg_data/Codellama-1Layer-1Image/checkpoint_12.pt 