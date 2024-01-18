import os
import json
import argparse
import importlib
from os.path import join
import torch 

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype
from open_flamingo.train.train_utils import validate_one_mujoco_epoch
from data import get_data
from tqdm import tqdm
from einops import rearrange
import numpy as np
from PIL import Image
import wandb
from glob import glob
from natsort import natsorted

"""
inspect eval output and upload to wandb
python real2code_inspect.py --checkpoint_path  /local/real/mandi/flg_data/datav2-test-emb/checkpoint_40.pt --no_vis_encoder 
"""

MESH_DATA_DIR = "/local/real/mandi/blender_dataset_v2"

OBB_HELPER="""def compute_joint_from_obb(obb_edge, axis_idx, direction, obb):  
    obb_center = np.array(obb['center'])
    obb_R = np.array(obb['R'])
    obb_half_lengths = np.array(obb['half_lengths']) 
    obb_edge = np.insert(np.array(obb_edge), axis_idx, 0)
    joint_pos = obb_center + obb_R @ (obb_edge * obb_half_lengths)
    obb_axis = obb_R[:, axis_idx]
    joint_axis = np.array(obb_axis) * direction  
    return joint_pos.tolist(), joint_axis.tolist()
"""

CODE_HEADER = [
    "from dm_control import mjcf",
    "import mujoco",
    "from PIL import Image",
    "import numpy as np",
    OBB_HELPER,
]

VIS_CODE = [
        "cam = model.worldbody.add('camera', name='camera', pos=[-2, 3, 3], mode='targetbody', target='root', fovy=50)",
        "mjcf_physics = mjcf.Physics.from_mjcf_model(model)",

        "mjcf_physics.model._model.vis.global_.offwidth = 800",
        "mjcf_physics.model._model.vis.global_.offheight = 800",
        "mjcf_physics.model._model.vis.headlight.ambient = [0.8,0.8,0.8]",
        "mjcf_physics.model._model.vis.headlight.diffuse = [0.8,0.8,0.8]",
        "mjcf_physics.model._model.vis.headlight.specular = [0.8,0.8,0.8]",

        "low = mjcf_physics.model.jnt_range[:, 0]",
        "high = mjcf_physics.model.jnt_range[:, 1]", 
        "val = np.random.uniform(low=low, high=high)",
        "mjcf_physics.data.qpos[:] = high",
        "mjcf_physics.data.qvel[:] = 0",
        "mjcf_physics.forward()",
        "img = mjcf_physics.render(camera_id='camera', height=800, width=800, depth=False)",
        "img = Image.fromarray(img)",
]
def execute_predictions(input_text, prediction, output_dir, mesh_folder="blender_meshes"): 
    pred_img_name = join(output_dir, 'pred_img.jpg')
    preds = prediction.split("\n")
    adjusted_preds = []
    for line in preds:
        if not (line.startswith("body") or line.startswith("joint")):
            continue
        elif ".add('body', " in line:
            # add inertial
            body_name = line.split("=")[0].strip()  
            iner_line = f"{body_name}.add('inertial', pos=[0, 0, 0], mass=1, diaginertia=[1, 1, 1])"
            if line[:4] == "    ":
                iner_line = "    " + iner_line # for the indentation 
            adjusted_preds.extend([line, iner_line])  
        else:
            adjusted_preds.append(line)
    if len(adjusted_preds) == 0:
        print(f"No valid prediction is found in \n{output_dir}")
        return None 
    adjusted_input = []
    for line in input_text.split("\n"):
        if line.startswith("model.asset.add("):
            refline = line.replace(")", ", refquat='0.50 -0.5 0.5 0.5')")
            adjusted_input.append(refline) 
        elif "<image>" in line or "<s>" in line or "from utils import" in line:
            continue
        elif line.startswith("model = mjcf"):
            extended = [
                line,
                "model.compiler.autolimits = 'true'",
                "model.compiler.angle = 'radian'", 
                f"model.compiler.meshdir = '{mesh_folder}'"
            ]
            adjusted_input.extend(extended)
        elif line.startswith("body_root = model.worldbody.add("):
            adjusted_input.extend([
                line, f"body_root.add('inertial', pos=[0, 0, 0], mass=1, diaginertia=[1, 1, 1])"])
        else:
            adjusted_input.append(line)
    
    starter_code = CODE_HEADER.copy() + adjusted_input
    starter_code.extend(adjusted_preds)
    starter_code.extend(VIS_CODE)
    starter_code.append(f"img.save('{pred_img_name}')")
    
    code_header = "\n".join(starter_code)
    
    # save as python file
    with open(join(output_dir, 'pred.py'), 'w') as f:
        f.write(code_header)
    try:
        exec(code_header) 
        output_img = Image.open(pred_img_name) 
        print(f"Code executed. Saved {pred_img_name}")
    except Exception as e:
        print(f"Code execution failed: {e}")
        output_img = None
    return output_img   

def run(args, ckpt_dir):
    lookup_path = join(ckpt_dir, args.lookup_type, args.lookup_folder)
    output_dirs = natsorted(glob(lookup_path))
    columns = ["obj_type", "obj_folder", "input", "image", "loss", "pred", "label"]
    if args.try_exec_pred:
        columns.append("output_image")
    table = wandb.Table(columns=columns)
    exec_success = 0
    num_rows = 0
    for output_dir in output_dirs:
        obj_type = output_dir.split('/')[-2]
        obj_folder = output_dir.split('/')[-1]
        pred_file = join(output_dir, "predictions.json")
        if not os.path.exists(pred_file):
            continue
        for fname in ["input.json", "label.json", "loss.json"]:
            assert os.path.exists(join(output_dir, fname)), f"{fname} not found in {output_dir}"

        input_text = json.load(open(join(output_dir, 'input.json'), 'r'))
        if isinstance(input_text, list):
            input_text = input_text[0]
        loss  = json.load(open(join(output_dir, 'loss.json'), 'r'))
        input_imgs = glob(join(output_dir, '*.png'))
        # concat images
        pil_imgs = [Image.open(img) for img in input_imgs]
        np_imgs = [np.array(img) for img in pil_imgs]
        concat_img = np.concatenate(np_imgs, axis=1)
        concat_img = Image.fromarray(concat_img).convert('RGB')
        wandb_img = wandb.Image(concat_img)
        pred = json.load(open(pred_file, 'r'))
        
        label = json.load(open(join(output_dir, 'label.json'), 'r'))
        
        row = [obj_type, obj_folder,  input_text, wandb_img, loss, pred, label] 
        if args.try_exec_pred:
            mesh_folder = join(MESH_DATA_DIR, "test", obj_type, obj_folder, "blender_meshes")
            assert os.path.exists(mesh_folder), f"{mesh_folder} not found"
            output_img = execute_predictions(input_text, pred,output_dir,  mesh_folder)
            if output_img is None:
                output_img = Image.new('RGB', (64,64))
            else:
                exec_success += 1    
            row.append(wandb.Image(output_img))
            
        table.add_data(*row)
        num_rows += 1
        print(f"Row: {num_rows}, Added: {output_dir}")

    run_name = "\n".join(ckpt_dir.split("/")[-2:])
    run_name += f"_success_{exec_success}_total_{num_rows}"
    print(f"Done processing: {run_name}")
    if num_rows > 0 and args.wandb: 
        wandb.init(project="real2code", name=run_name, group="inspect")
        wandb.log({"Outputs": table})
        wandb.finish()
    return 

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
    parser.add_argument("--try_exec_pred", "-try", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_vis_encoder", action="store_true")
    parser.add_argument("--is_val", action="store_true", default=True)

    parser.add_argument("--lookup_type", type=str, default="*")
    parser.add_argument("--lookup_folder", type=str, default="*") # or change to a specific folder name
    args = parser.parse_args()
    
    run_name = os.path.dirname(args.checkpoint_path).split('/')[-1] 
    ckpt_name = os.path.basename(args.checkpoint_path).split('.')[0]
    ckpt_dir = join(args.eval_data_dir, run_name, ckpt_name)
      
    run(args, ckpt_dir)