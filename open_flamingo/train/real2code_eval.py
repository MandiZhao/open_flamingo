import os
import json
import argparse
import importlib
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype
from data import get_data
from tqdm import tqdm
from einops import rearrange

"""
export WORLD_SIZE=3
export WDS_CACHE=/local/real/mandi/model_cache
torchrun --nnodes=1 --nproc_per_node=3  real2code_eval.py
"""

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    arg_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    if not os.path.exists(arg_path):
        print('Training args not saved, filling with default values')
        model_args = dict(
            vision_encoder_path="ViT-L-14",
            lm_path="/home/mandi/codellama/CodeLlama-7b",
            lm_tokenizer_path="/home/mandi/codellama/CodeLlama-7b",
            cross_attn_every_n_layers=8,
            vision_encoder_pretrained="openai",
            precision="fp32",
            # device=-1, put on cpu first 
            cache_dir="/local/real/mandi/model_cache",
            checkpoint_path=checkpoint_path,
            )
    else:
        with open(arg_path, "r") as f:
            model_args = json.load(f)
    module = importlib.import_module(f"open_flamingo.eval.models.open_flamingo")
    eval_model = module.EvalModel(model_args)
    return eval_model

def main(args):
    args.model = "open_flamingo"
    args.fsdp_use_orig_params = True 
    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    if args.rank == 0:
        print(f"Running on {args.world_size} GPUs. Loading Checkpoint") 
    
    eval_model = load_model(args.checkpoint_path)
    mp_policy = None
    args.my_group = None  # for optimizer saving
    process_group = None  # for FSDP init
    # init FSDP
    wrapper_kwargs = dict(
        process_group=process_group,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=device_id,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=ShardingStrategy.FULL_SHARD
        if args.fsdp_sharding_strategy == "full"
        else ShardingStrategy.HYBRID_SHARD,
        use_orig_params=args.fsdp_use_orig_params,
        mixed_precision=mp_policy,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
    # start wrapping model
    eval_model.model.wrap_fsdp(wrapper_kwargs, device_id)
    # eval_model.init_distributed() skip this since using fsdp
    
    dataset_args = dict(
        mujoco_shards="/local/real/mandi/mobility_shards/train/000000000.tar",
        dataset_resampled=True,
        train_num_samples_mujoco=418,
        seed=42,
        batch_size_mujoco=4,
        workers=4,
    )
    for k, v in dataset_args.items():
        setattr(args, k, v)
    dataset = get_data(args, eval_model.image_processor, eval_model.tokenizer, "mujoco")
    loader = dataset.dataloader

    num_eval_steps = 1
    cast_dtype = None
    for step, batch in tqdm(
        enumerate(loader),
        disable=(args.rank != 0),
        total=num_eval_steps,
    ):
        print('Preparing Input')
        images = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch[2].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        print(eval_model.tokenizer.decode(input_ids[0]))
        eos_token_id = eval_model.tokenizer.encode("<|endofchunk|>")[-1]
        with torch.inference_mode():
            with eval_model.autocast():
                # print('Try forwarding LM')
                # out = eval_model.model.forward(images, input_ids,  attention_mask=attention_mask)

                print('Generating')
                outputs = unwrap_model(eval_model.model).generate(
                    images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=10,
                    max_new_tokens=128,
                    num_beams=1,
                    length_penalty=0.0,
                    eos_token_id=eos_token_id,
                )
        outputs = outputs[:, len(input_ids[0]) :]
        print('Decoding')
        predictions = eval_model.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        print(predictions)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="open_flamingo")
    parser.add_argument("--checkpoint_path", type=str, default="/home/mandi/open_flamingo/open_flamingo/train/test-Codellama/checkpoint_25.pt")
    parser.add_argument("--horovod", action="store_true")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full")
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--no-set-device-rank", default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    
    args = parser.parse_args()
    main(args)

