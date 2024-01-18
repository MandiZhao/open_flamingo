# export NCCL_SOCKET_IFNAME=enp37s0f0
export MASTER_ADR=25678
export WDS_CACHE_SIZE=800
export WDS_CACHE=/local/real/mandi/model_cache
export WORLD_SIZE=3
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE}  --master_port=25678 real2code_eval.py --checkpoint_path  /local/real/mandi/flg_data/datav2-test-emb/checkpoint_40.pt --no_vis_encoder 
