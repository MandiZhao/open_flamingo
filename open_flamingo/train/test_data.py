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

import functools
from open_flamingo import create_model_and_transforms
import webdataset as wds
import os
import io
from PIL import Image
import base64

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def run(args):
    image_processor, tokenizer = None, None
    # model, image_processor, tokenizer = create_model_and_transforms(
    #     "ViT-L-14",
    #     "openai",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     "/home/mandi/codellama/CodeLlama-7b",
    #     cross_attn_every_n_layers=2,
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
            img_names = [name for name in image_names if f"camera_{i}" in name]
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
        print([b.shape for b in batch])
        breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco_shards", type=str, default="/local/real/mandi/mobility_shards/train/000000000.tar")
    parser.add_argument("--dataset_resampled", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_num_samples_mujoco", type=int, default=418)
    parser.add_argument("--batch_size_mujoco", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    run(args)

