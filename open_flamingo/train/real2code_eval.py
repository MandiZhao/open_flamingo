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
    print('Wrapping model')
    eval_model.model.wrap_fsdp(wrapper_kwargs, device_id)
    tokenizer = eval_model.tokenizer
    image_processor = eval_model.image_processor
    # eval_model.init_distributed() skip this since using fsdp
    print('Loading dataset')
    dataset_args = dict(
        mujoco_shards="/local/real/mandi/mobility_shards/val/000000000.tar",
        dataset_resampled=1,
        train_num_samples_mujoco=51,
        seed=42,
        batch_size_mujoco=1,
        workers=1,
    )
    for k, v in dataset_args.items():
        setattr(args, k, v)
    args.is_val = True # cut the input to be short
    dataset = get_data(args, eval_model.image_processor, eval_model.tokenizer, "mujoco")
    loader = dataset.dataloader

    cast_dtype = None
    with torch.inference_mode():
        with eval_model.autocast():
            media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1] 
            eos_token_id = tokenizer.encode("<|endofchunk|>")[-1]

    for step, batch in tqdm(
        enumerate(loader),
        disable=(args.rank != 0), 
    ):
        step_output_dir = os.path.join(output_dir, f"step_{step}")
        os.makedirs(step_output_dir, exist_ok=True)

        print('Preparing Input')
        images = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        # save images
        for t in range(images.shape[1]):
            img = images[0, t, 0].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(step_output_dir, f"image_{t}.png"))

        input_ids = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch[2].to(
            device_id, dtype=cast_dtype, non_blocking=True
        ) 
        label_text = batch[-1][0] # str
        # save label to json
        with open(os.path.join(step_output_dir, "label.json"), "w") as f:
            json.dump(label_text, f)

        with torch.inference_mode():
            with eval_model.autocast(): 
                decoded_input = tokenizer.batch_decode(
                            input_ids, skip_special_tokens=False)
                # if args.rank == 0:
                #     print("Input:")
                #     print(decoded_input)
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
                print("Validation Loss:", np.mean(loss.item()))
                # save loss to json
                with open(os.path.join(step_output_dir, "loss.json"), "w") as f:
                    json.dump(np.mean(loss.item()), f)
                if np.mean(loss.item()) > 10:
                    print("Loss is too high, skipping generation")
                    continue 
                
                outputs = unwrap_model(eval_model.model).generate(
                    images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=10,
                    max_new_tokens=1024,
                    num_beams=3,
                    length_penalty=0.0,
                    eos_token_id=eos_token_id,
                )

        outputs = outputs[:, len(input_ids[0]) :] 
        predictions = eval_model.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        if args.rank == 0:
            print("Output:")
            print(predictions[0])
        # save predictions to json
        with open(os.path.join(step_output_dir, "predictions.json"), "w") as f:
            json.dump(predictions[0], f)


def execute_predictions(args, output_dir, log_wandb=False):
    if log_wandb:
        wandb.init(project="real2code", name=os.path.basename(output_dir))
        table = wandb.Table(columns=["step", "input_text", "input_image", "loss", "output_text", "output_image"])
    steps = [s for s in os.listdir(output_dir) if s.startswith('step_')]
    for step in steps:
        pred_img_name = join(output_dir, step, 'pred_img.jpg')
        input_text = json.load(open(join(output_dir, step, 'input.json'), 'r'))
        if isinstance(input_text, list):
            input_text = input_text[0]
        loss  = json.load(open(join(output_dir, step, 'loss.json'), 'r'))
        input_img = join(output_dir, step, 'image_0.png')
        input_img = wandb.Image(input_img)
        
        row = [int(step.split('_')[-1]), input_text, input_img, loss, "", None]
        if os.path.exists(join(output_dir, step, 'predictions.json')):
            preds = json.load(open(join(output_dir, step, 'predictions.json'), 'r'))
            row[4] = preds
            preds = preds.split("\n")
            code_header = [
                "from dm_control import mjcf",
                "import mujoco",
                "from PIL import Image",
                "model = mjcf.RootElement(model='object')",
                "model.compiler.autolimits = 'true'",
                "model.compiler.angle = 'radian'",
            ]
            code_header.extend(preds)
            code_header.extend([
                "cam = model.worldbody.add('camera', name='camera', pos=[-2, 3, 3], mode='targetbody', target='object', fovy=30)",
                "mjcf_physics = mjcf.Physics.from_mjcf_model(model)",
                "low = mjcf_physics.model.jnt_range[:, 0]",
                "high = mjcf_physics.model.jnt_range[:, 1]",
                "import numpy as np",
                "val = np.random.uniform(low=low, high=high)",
                "mjcf_physics.data.qpos[:] = high",
                "mjcf_physics.data.qvel[:] = 0",
                "mjcf_physics.forward()",
                "img = mjcf_physics.render(camera_id='camera', height=480, width=480, depth=False)",
                "img = Image.fromarray(img)",
                f"img.save('{pred_img_name}')",
            ])
            code_header = "\n".join(code_header)
            # save as python file
            with open(join(output_dir, step, 'pred.py'), 'w') as f:
                f.write(code_header)
            try:
                exec(code_header)
                print(f"Code executed. Saved {pred_img_name}")
                output_img = wandb.Image(pred_img_name)
                row[5] = output_img
            except Exception as e:
                print(f"Code execution failed: {e}")
        if log_wandb:
            table.add_data(*row)
    if log_wandb:
        wandb.log({"Outputs": table})
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="open_flamingo")
    parser.add_argument("--checkpoint_path", type=str, default="/home/mandi/open_flamingo/open_flamingo/train/test-Codellama/checkpoint_25.pt")
    parser.add_argument("--eval_data_dir", type=str, default="/home/mandi/flg_share/eval_data")
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
    args = parser.parse_args()
    
    run_name = os.path.dirname(args.checkpoint_path).split('/')[-1] 
    ckpt_name = os.path.basename(args.checkpoint_path).split('.')[0]
    output_dir = join(args.eval_data_dir, run_name, ckpt_name)
    
    if args.try_exec_pred:
        execute_predictions(args, output_dir, args.wandb)
        exit()

    main(args, output_dir)

