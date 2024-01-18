import os
import json
import argparse
import importlib
from os.path import join
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
from open_flamingo.train.train_utils import validate_one_mujoco_epoch
from data import get_data
from tqdm import tqdm
from einops import rearrange
import numpy as np
from PIL import Image
import wandb

"""
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WORLD_SIZE=6
export WDS_CACHE=/local/real/mandi/model_cache
torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} real2code_eval.py --checkpoint_path /home/mandi/flg_share/test-Codellama-12Layer/checkpoint_4.pt
"""

def load_model(args, checkpoint_path):
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
            no_vis_encoder=args.no_vis_encoder,
            )
    else:
        with open(arg_path, "r") as f:
            model_args = json.load(f)
        model_args["checkpoint_path"] = checkpoint_path
        model_args["device"] = -1
        if "lm_tokenizer_path" not in model_args:
            model_args["lm_tokenizer_path"] = model_args["lm_path"]
    # print args with indent:
    print(json.dumps(model_args, indent=4))
    module = importlib.import_module(f"open_flamingo.eval.models.open_flamingo")
    eval_model = module.EvalModel(model_args)
    return eval_model

def main(args, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    args.model = "open_flamingo"
    args.fsdp_use_orig_params = True 
    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)

    # eval_model.init_distributed() skip this since using fsdp
    
    if args.rank == 0:
        print(f"Running on {args.world_size} GPUs. Loading Checkpoint") 
    
    eval_model = load_model(args, args.checkpoint_path)
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
    print('Wrapping model')
    eval_model.model.wrap_fsdp(wrapper_kwargs, device_id)
    tokenizer = eval_model.tokenizer
    image_processor = eval_model.image_processor
    
    print('Loading dataset')
    dataset_args = dict(
        mujoco_shards=args.eval_shard,
        dataset_resampled=1,
        train_num_samples_mujoco=160,
        seed=42,
        batch_size_mujoco=1,
        workers=1,
        is_val=True,
        no_vis_encoder=args.no_vis_encoder,
    )
    for k, v in dataset_args.items():
        setattr(args, k, v)
    args.is_val = True # cut the input to be short
    dataset = get_data(args, eval_model.image_processor, eval_model.tokenizer, "mujoco", is_val=True)
    loader = dataset.dataloader 
    

    cast_dtype = None
    # with torch.inference_mode():
    with torch.no_grad():
        with eval_model.autocast():
            media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1] 
            eos_token_id = tokenizer.encode("<|endofchunk|>")[-1]

    for step, batch in tqdm(
        enumerate(loader),
        disable=(args.rank != 0), 
    ):
        assert len(batch) == 4, "Batch should be (images, input_ids, attention_mask, meta_data)"
        meta_data = batch[3][0] # on each rank, len(batch) == batch_size_mujoco
        label_text = meta_data['text']
        obj_type = meta_data['obj_type']
        obj_folder = meta_data['obj_folder']
        obj_dir = meta_data['obj_dir']
        
        step_output_dir = os.path.join(output_dir, obj_type, obj_folder)
        if not os.path.exists(step_output_dir):
            os.makedirs(step_output_dir)
        else:
            # skip if already generated
            # print(f"Skipping {step_output_dir} since already generated")
            continue
        image_names = meta_data['image_names']
        for img_name in image_names: # e.g. ['loop_0_rgb_8']
            if 'rgb' not in img_name:
                continue
            loop_id = img_name.split('_')[1]
            rgb_id = img_name.split('_')[-1] 
            full_img_path = os.path.join(obj_dir, f"loop_{loop_id}", f"rgb_{rgb_id}.png")
            if not os.path.exists(full_img_path):
                print(f"Image {full_img_path} does not exist, skipping")
                continue
            img = Image.open(full_img_path)
            # save to output dir
            img.save(join(step_output_dir, f"{img_name}.png"))

        if args.no_vis_encoder:
            images = batch[0].to(device_id)
        else:
            images = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
            images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
            # save images
            for t in range(images.shape[1]):
                img = images[0, t, 0].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img).convert("RGB")
                img.save(os.path.join(step_output_dir, f"image_{t}.png"))

        input_ids = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch[2].to(
            device_id, dtype=cast_dtype, non_blocking=True
        ) 
        
        # save label to json 
        with open(os.path.join(step_output_dir, "label.json"), "w") as f:
            json.dump(label_text, f)

        with open(os.path.join(step_output_dir, "meta.json"), "w") as f:
            json.dump(meta_data, f)

        # with torch.inference_mode():
        with torch.no_grad():
            with eval_model.autocast(): 
                decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=False) 
                # save decoded str to json
                with open(os.path.join(step_output_dir, "input.json"), "w") as f:
                    json.dump(decoded_input, f)
                # print('Try forwarding LM')
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                labels[labels == tokenizer.eos_token] = -100
                labels[labels == media_token_id] = -100
                labels = labels.to(device_id)

                with torch.no_grad(): 
                    loss = eval_model.model(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )[0]
                labels.detach()
                # if args.rank == 0:
                #     print("Validation Loss:", np.mean(loss.item()))
                # save loss to json
                with open(os.path.join(step_output_dir, "loss.json"), "w") as f:
                    json.dump(np.mean(loss.item()), f)
                
                # if np.mean(loss.item()) > 10:
                #     print("Loss is too high, skipping generation")
                #     continue 
                
                outputs = unwrap_model(eval_model.model).generate(
                    images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=10,
                    max_new_tokens=512,
                    num_beams=3,
                    length_penalty=0.0,
                    eos_token_id=eos_token_id,
                    temperature=0,
                )

        outputs = outputs[:, len(input_ids[0]) :] 
        predictions = eval_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred = predictions[0]
        break_str = "body_root = model.worldbody.add('body', name='root')\n"
        label = label_text.split(break_str)[1]
        toprint = f"===== Input =====\n{decoded_input[0]}\n=====\n===== Output =====\n{pred}\n=====\n===== Label =====\n{label}\n=====\n"
        print(f"Rank {args.rank}:\n{toprint}")

        # save predictions to json
        with open(os.path.join(step_output_dir, "predictions.json"), "w") as f:
            json.dump(pred, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="open_flamingo")
    parser.add_argument("--checkpoint_path", type=str, default="/home/mandi/open_flamingo/open_flamingo/train/test-Codellama/checkpoint_25.pt")
    parser.add_argument("--eval_data_dir", type=str, default="/home/mandi/flg_share/eval_data")
    parser.add_argument("--eval_shard", type=str, default="/local/real/mandi/mobility_shards_v2_emb/val/0000.tar")
    parser.add_argument("--mujoco_val_shards", type=str, default="/local/real/mandi/mobility_shards_v2_loop_0_emb/val/0000.tar")
    
    parser.add_argument("--val_num_samples_mujoco", type=int, default=160)

    parser.add_argument("--horovod", action="store_true")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full")
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--no-set-device-rank", default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--try_exec_pred", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_vis_encoder", action="store_true")
    parser.add_argument("--is_val", action="store_true", default=True)
    args = parser.parse_args()
    
    run_name = os.path.dirname(args.checkpoint_path).split('/')[-1] 
    ckpt_name = os.path.basename(args.checkpoint_path).split('.')[0]
    output_dir = join(args.eval_data_dir, run_name, ckpt_name)
      
    main(args, output_dir)

