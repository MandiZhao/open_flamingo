import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_data, get_mujoco_dataset, preprocess_mujoco_text, preprocess_image
from data_utils import * 

from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype

import functools
from open_flamingo import create_model_and_transforms
import open_clip 
import webdataset as wds
import os
import io
from PIL import Image
import base64
from transformers import SamModel, SamProcessor

"""
test fsdp:
 
export WDS_CACHE_SIZE=800
export WDS_CACHE=/local/real/mandi/model_cache
export WORLD_SIZE=3
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} test_training.py --no_vis_encoder --test_fsdp
"""
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def test_model(args):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        cache_dir="/local/real/mandi/model_cache",
    )
    sam_encoder = SamModel.from_pretrained("facebook/sam-vit-huge")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # compare parameter count:
    count1 =  sum([p.numel() for p in vision_encoder.parameters()])
    count2 = sum([p.numel() for p in sam_encoder.parameters()])
    sam_base_encoder = SamModel.from_pretrained("facebook/sam-vit-base")
    count3 = sum([p.numel() for p in sam_base_encoder.parameters()])
                  
    print("vision encoder", count1)
    print("sam encoder", sum([p.numel() for p in sam_encoder.parameters()]))
    breakpoint()


def run(args):
    # test_model(args)
    # exit()
    image_processor, tokenizer = None, None
    # model, image_processor, tokenizer = create_model_and_transforms(
    #     "ViT-L-14",
    #     "openai",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     cross_attn_every_n_layers=16q,
    #     use_local_files=True,
    #     gradient_checkpointing=False,
    #     freeze_lm_embeddings=True,
    #     cache_dir="/local/real/mandi/model_cache",
    # )
    
    def process_images_and_text(inp, np_seed=0, num_cameras=4):
        # img1, img2, img3, img4, json = inp
        # stacked = [img1, img2, img3, img4]
        # images = preprocess_image(stacked, image_processor)
        
        # ids, mask = preprocess_mujoco_text(
        #     [json["text"]], 
        #     tokenizer)
        # ids = ids.squeeze(0)
        # mask = mask.squeeze(0) # squeeze from (batch_size, 1, seq_len)
        # print(ids.shape, mask.shape)
        # return images, ids, mask

        inp = inp[0] 
        image_names = inp["images"].keys()
        np_random = np.random.RandomState(np_seed)
        sampled_names = []
        for i in range(num_cameras):
            img_names = [name for name in image_names if f"camera_{i}" in name and "seg" in name]
            sampled_names.append(np_random.choice(img_names))
        images = []
        for k in sampled_names:
            base_str = inp["images"][k]
            rawbytes = base64.b64decode(base_str)
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            images.append(image)
        images = preprocess_image(images, image_processor)
        ids, mask = preprocess_mujoco_text(
            [inp["text"]], 
            tokenizer)
        ids = ids.squeeze(0)
        mask = mask.squeeze(0) 
        # print("before token", inp["text"])
        # print("after token", tokenizer.decode(ids))
        return images, ids, mask
 
    # dataset = get_data(args, image_processor, tokenizer, dataset_type="mujoco", epoch=0)
    shared_epoch = SharedEpoch(epoch=0)
    pipeline = [wds.SimpleShardList(args.mujoco_shards)]
    pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                tarfile_to_samples_nothrow,
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
                # wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                # wds.to_tuple("camera_0_0_seg.jpg", "camera_1_0_seg.jpg", "camera_2_0_seg.jpg", "camera_3_0_seg.jpg", "json"),
                wds.to_tuple("json"),
                wds.map(process_images_and_text),  
                wds.batched(args.batch_size_mujoco, partial=False),  
            ]
        )
    dataset = wds.DataPipeline(*pipeline)
    dataset = dataset.with_epoch(3)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    for i, batch in enumerate(dataloader):
        images = batch[0]
        step_output_dir = os.path.join("test_output", f"step_{i}")
        os.makedirs(step_output_dir, exist_ok=True)
        for t in range(images.shape[1]):
            img = images[0, t].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            img.save(os.path.join(step_output_dir, f"image_{t}.png")) 
        # breakpoint()

def examine_data(args):
    if args.test_fsdp:
        args.fsdp_use_orig_params = True 
        # set up distributed evaluation
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        device_id = init_distributed_device(args)

    image_processor, tokenizer = None, None
    # model, image_processor, tokenizer = create_model_and_transforms(
    #     "ViT-L-14",
    #     "openai",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     cross_attn_every_n_layers=16,
    #     use_local_files=True,
    #     gradient_checkpointing=False,
    #     freeze_lm_embeddings=True,
    #     cache_dir="/local/real/mandi/model_cache",
    #     no_vis_encoder=args.no_vis_encoder,
    # )
    dataset_args = dict(
        fsdp_sharding_strategy="full",
        fsdp_use_orig_params=True,
        mujoco_shards=args.mujoco_shards,
        dataset_resampled=1,
        train_num_samples_mujoco=10000,
        seed=42,
        batch_size_mujoco=1,
        workers=1,
        is_val=1,
        no_vis_encoder=args.no_vis_encoder,
    )
    for k, v in dataset_args.items():
        setattr(args, k, v)
    dataset = get_data(args, image_processor, tokenizer, "mujoco", is_val=True)
    loader = dataset.dataloader
    if args.test_fsdp:
        print(f"Rank: {args.rank}, world_size: {args.world_size} | Loader num batches: {loader.num_batches}")
   
    output_dir = "test_output"
    for i, batch in enumerate(loader):
        images = batch[0]
        # step_output_dir = os.path.join("test_output", f"step_{i}")
        # os.makedirs(step_output_dir, exist_ok=True)
        # for t in range(images.shape[1]):
        #     img = images[0, t].cpu().numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = (img * 255).astype(np.uint8)
        #     img = Image.fromarray(img).convert("RGB")
        #     img.save(os.path.join(step_output_dir, f"image_{t}.png"))
        assert len(batch) == 4
        meta_data = batch[-1]
        # print(len(meta_data)) on each rank, len(batch) == batch_size_mujoco
        meta_data = meta_data[0]
        obj_type = meta_data["obj_type"]
        obj_id = meta_data["obj_folder"]
        img_names = meta_data["image_names"]
        print(f"Rank: {args.rank}, world_size: {args.world_size} | Obj type: {obj_type} | Obj id: {obj_id} | Images: {img_names}")
        # save obj type and folder to file
        fname = os.path.join(output_dir, f"obj_type_{obj_type}_obj_id_{obj_id}.txt")
        # assert not os.path.exists(fname), f"File {fname} already exists!"
        # with open(fname, "w") as f:
        #     f.write("test")
         
    exit()
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco_shards", type=str, default="/local/real/mandi/mobility_shards_v2_emb_loop_0/val/0000.tar")
    parser.add_argument("--mujoco_val_shards", type=str, default="/local/real/mandi/mobility_shards_v2_loop_0_emb/val/0000.tar")
    parser.add_argument("--val_num_samples_mujoco", type=int, default=160)

    parser.add_argument("--dataset_resampled", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_num_samples_mujoco", type=int, default=1977)
    parser.add_argument("--batch_size_mujoco", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--no_vis_encoder", action="store_true", default=False)

    parser.add_argument("--test_fsdp", action="store_true", default=False)
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
    examine_data(args)
    # run(args)

