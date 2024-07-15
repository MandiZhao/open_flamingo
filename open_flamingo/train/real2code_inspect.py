import os
import json
import argparse
import importlib
from os.path import join
import torch 
from copy import deepcopy
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
from collections import defaultdict
"""
New v3 data: must load the vis code from saved data, train/test data doesn't contain model.asset.add code anymore
inspect eval output and upload to wandb
DATADIR=/local/real/mandi/flg_data
MODEL=datav2-short-1024-4ly/checkpoint_8.pt
python real2code_inspect.py --model ${MODEL} --shorten_text --gen_type top_p_p0.85_t0.2


MODEL=datav3-1645-101ly/checkpoint_14.pt 
python real2code_inspect.py --model ${MODEL} --gen_type "*"
"""


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
COMPRESS_HELPER="""def add_body_and_joint(parent_body, mesh_id, obb, obb_axis_idx, obb_edge, obb_sign, joint_type):  
    center = np.array(obb['center'])
    obb_R = np.array(obb['R'])
    obb_half_lengths = np.array(obb['extents']) / 2
    obb_edge = np.insert(np.array(obb_edge), obb_axis_idx, 0)
    joint_pos = center + obb_R @ (obb_edge * obb_half_lengths)
    obb_axis = obb_R[:, obb_axis_idx]
    joint_axis = np.array(obb_axis) * obb_sign  
    
    link_body = parent_body.add('body', name=f'link_{mesh_id}')
    link_body.add('geom', type='mesh', name=f'link_{mesh_id}_geom', mesh=f'link_{mesh_id}')
    link_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
    link_body.add('joint', name=f'joint_{mesh_id}', pos=joint_pos, axis=joint_axis, type=joint_type, range=[0, 1])
    return link_body
"""
COMPRESSED_ADD_CODE = """
for joint in child_joints:
    box_id = joint['box']
    add_body_and_joint(body_root, box_id, bboxes[box_id], joint['idx'], joint['edge'], joint['sign'], joint['type'])
# add root body's geom:
body_root.add('geom', type='mesh', name='root_geom', mesh=f'link_{root_geom}')
"""
CODE_HEADER = [
    "from dm_control import mjcf",
    "import mujoco",
    "from PIL import Image",
    "import numpy as np",
    OBB_HELPER,
]

V3_SKIP_IDS=[47585, 45725] + [19898, 22241, 25493, 26652, 22367, 26608, 30666]
VIS_CODE = [
    "from PIL import Image",
    "cam = model.worldbody.add('camera', name='camera', pos=[-2, 3, 3], mode='targetbody', target='root', fovy=50)",
    "physics = mjcf.Physics.from_mjcf_model(model)",

    "physics.model._model.vis.global_.offwidth = 800",
    "physics.model._model.vis.global_.offheight = 800",
    "physics.model._model.vis.headlight.ambient = [0.8,0.8,0.8]",
    "physics.model._model.vis.headlight.diffuse = [0.8,0.8,0.8]",
    "physics.model._model.vis.headlight.specular = [0.8,0.8,0.8]",

    "low = physics.model.jnt_range[:, 0]",
    "high = physics.model.jnt_range[:, 1]", 
    "val = np.random.uniform(low=low, high=high)",
    "physics.data.qpos[:] = high",
    "physics.data.qvel[:] = 0",
    "physics.forward()",
    "img = physics.render(camera_id='camera', height=800, width=800, depth=False)",
    "img = Image.fromarray(img)",
]

def adjust_input(input_text, mesh_folder="blender_meshes"):
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
    return adjusted_input

def v3_get_exec_code(adjusted_preds, mesh_folder="blender_meshes", gt_data=dict()):
    """ Replace the GT label with model predictions, results should be directly executable """
    assert gt_data.get('vis_code', None) is not None, "vis_code is required for v3 data"
    assert gt_data.get('test_code', None) is not None, "test_code is required for v3 data"
    gt_label = gt_data['test_code'].split("root_geom = ")[1]
    
    vis_code = deepcopy(gt_data['vis_code'])
    if any([str(id) in mesh_folder for id in V3_SKIP_IDS]):
        vis_code = vis_code.replace("train", "test") # special case since these two folders were moved from train to test

    exec_code = vis_code.replace(gt_label, "\n".join(adjusted_preds)) 
    return exec_code

def adjust_prediction(prediction):
    """ remove the invalid lines"""
    preds = prediction.split("\n")
    adjusted_preds = []
    for line in preds:
        if not (line.startswith("body_") or line.startswith("joint_")):
            continue
        elif ".add('body', " in line:
            # add inertial
            body_name = line.split("=")[0].strip()  
            iner_line = f"{body_name}.add('inertial', pos=[0, 0, 0], mass=1, diaginertia=[1, 1, 1])"
            if line[:4] == "    ":
                iner_line = "    " + iner_line # for the indentation 
            adjusted_preds.extend([line, iner_line])  
        else:
            cutoff = line.split(")")[0] + ")"
            adjusted_preds.append(cutoff)
    return adjusted_preds

def v3_adjust_pred(prediction):
    preds = prediction.split("\n")
    adjusted_preds = []
    if "child_joints = " not in prediction:
        return adjusted_preds
    for line in preds:
        # get int value
        if len(line.strip()) <= 2 and line.strip().isdigit(): # handles single or double digits
            adjusted_preds.append(line) 
        if line.startswith("child_joints = ") or line.startswith("dict(box="):
            adjusted_preds.append(line)
        if line.startswith("]"):
            adjusted_preds.append(line.split("]")[0] + "]") 
        if 'child_joints = [' in adjusted_preds and "]" in adjusted_preds:
            break # early stop
    return adjusted_preds

def execute_predictions(args, input_text, prediction, output_dir, mesh_folder="blender_meshes", gt_data=None):
    pred_img_name = join(output_dir, 'pred_img.jpg')
    data_v3 = 'v3' in args.blender_data_dir or "v4" in args.blender_data_dir
    if data_v3:
        assert gt_data is not None, "gt_data is required for v3 data"
        adjusted_preds = v3_adjust_pred(prediction) 
    else:    
        adjusted_preds = adjust_prediction(prediction)
    if len(adjusted_preds) == 0:
        print(f"No valid prediction is found in \n{output_dir}")
        with open(join(output_dir, 'pred.py'), 'w') as f:
            f.write("\n") # still write an empty file!
        return None 
    if data_v3:
        if args.eval_shard:
            full_code = gt_data['text_info']['full_code'] 
            vis_code = "\n".join(VIS_CODE + [f"img.save('{pred_img_name}')"])
            full_code += "\n" + vis_code
            label_blanks = "0\nchild_joints = [\n]"
            code_header = full_code.replace(label_blanks, "\n".join(adjusted_preds))
            code_header = code_header.replace("extents", "extent")
        else:
            exec_code = v3_get_exec_code(adjusted_preds, mesh_folder, gt_data)
            code_header = exec_code.replace('test.png', pred_img_name)
    else:
        adjusted_input = adjust_input(input_text, mesh_folder)    
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
        print(f"Code execution in {output_dir} failed: {e}")
        output_img = None
    return output_img   

def pool_data_across_generations(args): 
    """ check for each object, how many of the generation runs returns decent predictions """
    run_name = os.path.dirname(args.model).split('/')[-1] 
    ckpt_name = os.path.basename(args.model).split('.')[0]
    gen_type = "*"
    lookup_dir = join(args.eval_data_dir, run_name, ckpt_name, gen_type, args.lookup_type, args.lookup_folder, "predictions.json")
    if args.eval_shard:
        lookup_dir = join(args.shard_data_dir, args.model, gen_type, args.lookup_type, args.lookup_folder, "predictions.json")
    pred_files = natsorted(glob(lookup_dir))
    # data from before using gen_type
    beam_dir = join(args.eval_data_dir, run_name, ckpt_name, args.lookup_type, args.lookup_folder, "predictions.json")
    pred_files.extend(natsorted(glob(beam_dir)))
    if len(pred_files) == 0:
        print(f"No output found in {lookup_dir}")
        return
    all_preds = defaultdict(list)
    all_files = defaultdict(list)
    all_imgs = defaultdict(list)
    table = wandb.Table(columns=["obj_type", "obj_folder", "input", "pred", "label", "gt_image", "pred_image"])
    for pred_file in pred_files:
        obj_type = pred_file.split('/')[-3]
        obj_folder = pred_file.split('/')[-2]  
        pred = json.load(open(pred_file, 'r'))
        adjusted_preds = adjust_prediction(pred)
        img_fname = pred_file.replace("predictions.json", "pred_img.jpg")
        name = f"{obj_type}_{obj_folder}"
        all_preds[name].append(adjusted_preds)
        all_files[name].append(pred_file) 

        if not os.path.exists(img_fname):
            all_imgs[name].append(None)
        else:
            all_imgs[name].append(wandb.Image(Image.open(img_fname)))
         
    for k, v in all_preds.items():
        num_valid = sum([img is not None for img in all_imgs[k]])
        print(f"{k}: valid predictions: {num_valid}/{len(v)}") 
        input_text = json.load(open(all_files[k][0].replace("predictions.json", "input.json"), 'r'))
        label_text = json.load(open(all_files[k][0].replace("predictions.json", "label.json"), 'r'))
        if isinstance(input_text, list):
            input_text = input_text[0]
        gt_image = glob(all_files[k][0].replace("predictions.json", "loop*.png"))
        if len(gt_image) > 0:
            gt_image = wandb.Image(gt_image[0])
        else:
            gt_image = wandb.Image(Image.new('RGB', (64,64)))
        obj_type, obj_folder = k.split("_")[0], k.split("_")[1]
        max_pred = v[0]
        max_img = wandb.Image(Image.new('RGB', (64,64)))
        if num_valid > 0:
            valid_idxs = [i for i, img in enumerate(all_imgs[k]) if img is not None]
            # select the longest prediction and its pred image file
            len_and_idx = [(len(v[i]), i) for i in valid_idxs]
            sorted_len_and_idx = sorted(len_and_idx, key=lambda x: x[0], reverse=True)
            max_idx = sorted_len_and_idx[0][1]
            max_pred = v[max_idx]
            max_img = all_imgs[k][max_idx]
        max_pred = "\n".join(max_pred)
            
        table.add_data(obj_type, obj_folder, input_text, max_pred, label_text, gt_image, max_img) 
    if args.wandb:
        wandb.init(project="real2code", name=f"pooled_{run_name}_{ckpt_name}", group="inspect")
        wandb.log({"Outputs": table})
        wandb.finish()
    # breakpoint()
    return all_preds

def run(args, ckpt_dir):
    lookup_path = join(ckpt_dir, args.lookup_type, args.lookup_folder)
    output_dirs = natsorted(glob(lookup_path))
    columns = ["obj_type", "obj_folder", "input", "image", "loss", "pred", "label"]
    if args.try_exec_pred:
        columns.extend(["output_image", "exec_success"])
    table = wandb.Table(columns=columns)
    exec_success = 0 
    all_rows = []
    for output_dir in output_dirs: 
        obj_type = output_dir.split('/')[-2]
        obj_folder = output_dir.split('/')[-1]
        pred_file = join(output_dir, "predictions.json")
        gt_data_fname = join(args.blender_data_dir, "test", obj_type, obj_folder, "obb_info_loop_0.json")
        if args.eval_shard:
            print("Not loading GT data")
            gt_data_fname = join(args.shard_data_dir, args.eval_sam_model, obj_type, obj_folder, "meshes/data_loop_0.json")
        if args.global_joints:
            gt_data_fname = join(args.blender_data_dir, "test", obj_type, obj_folder, "obb_info_loop_0_global.json")
        assert os.path.exists(gt_data_fname), f"{gt_data_fname} not found"
        with open(gt_data_fname, 'r') as f:
            gt_data = json.load(f)
        if not os.path.exists(pred_file):
            continue
        for fname in ["input.json", "label.json", "loss.json"]:
            assert os.path.exists(join(output_dir, fname)), f"{fname} not found in {output_dir}"

        input_text = json.load(open(join(output_dir, 'input.json'), 'r'))
        if args.shorten_text:
            # need the original input text to execute the prediction
            # data_dir = join(args.blender_data_dir, "test", obj_type, obj_folder)
            assert isinstance(input_text, dict) and "raw_input" in input_text, f"input_text is not a dict: {input_text}"
            input_text = input_text["raw_input"]
        if isinstance(input_text, list):
            input_text = input_text[0]
        loss  = json.load(open(join(output_dir, 'loss.json'), 'r'))
        input_imgs = glob(join(output_dir, '*.png'))
        # concat images
        pil_imgs = [Image.open(img) for img in input_imgs]
        # resize all to (480, 480)
        pil_imgs = [img.resize((480,480)) for img in pil_imgs]
        np_imgs = [np.array(img) for img in pil_imgs]
        concat_img = np.concatenate(np_imgs, axis=1)
        concat_img = Image.fromarray(concat_img).convert('RGB')
        wandb_img = wandb.Image(concat_img)
        pred = json.load(open(pred_file, 'r'))
        
        label = json.load(open(join(output_dir, 'label.json'), 'r'))
        
        row = [obj_type, obj_folder, input_text, wandb_img, loss, pred, label] 
        if args.try_exec_pred:
            mesh_folder = join(args.mesh_data_dir, "test", obj_type, obj_folder, "blender_meshes")
            assert os.path.exists(mesh_folder), f"{mesh_folder} not found"
            output_img = execute_predictions(args, input_text, pred, output_dir,  mesh_folder, gt_data)
            execed = output_img is not None
            if output_img is None:
                output_img = Image.new('RGB', (64,64))
            exec_success += execed   
            row.extend([wandb.Image(output_img), execed])
            
            all_rows.append(row)

    success_rows = defaultdict(list)
    for row in all_rows:
        obj_folder = row[1]
        if len(success_rows[obj_folder]) == 0:
            success_rows[obj_folder] = row
        else:
            if row[-1] > success_rows[obj_folder][-1]:
                success_rows[obj_folder] = row
    for k, v in success_rows.items():
        table.add_data(*v)
    run_name = "\n".join(ckpt_dir.split("/")[-2:])
    run_name += f"_success_{exec_success}_total_{len(all_rows)}"
    print(f"Done processing: {run_name}")
    if len(success_rows) > 0 and args.wandb: 
        wandb.init(project="real2code", name=run_name, group="inspect")
        wandb.log({"Outputs": table})
        wandb.finish()
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_data_dir", type=str, default="/local/real/mandi/blender_dataset_v3")
    parser.add_argument("--mesh_data_dir", type=str, default="/local/real/mandi/blender_dataset_v3") # NOTE eval on v3 data!
    parser.add_argument("--model", type=str, default="open_flamingo")  
    parser.add_argument("--eval_data_dir", type=str, default="/home/mandi/flg_share/eval_data") 
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--try_exec_pred", "-try", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_vis_encoder", action="store_true")
    parser.add_argument("--is_val", action="store_true", default=True)
    parser.add_argument("--shorten_text", action="store_true")
    parser.add_argument("--lookup_type", type=str, default="*")
    parser.add_argument("--lookup_folder", type=str, default="*") # or change to a specific folder name
    parser.add_argument("--gen_type", type=str, default="*") # or change to a specific folder name
    parser.add_argument("--pool_data", "-p", action="store_true")

    # for full eval:
    parser.add_argument("--eval_shard", "-es", default=False, action="store_true")
    parser.add_argument("--shard_data_dir", default="/home/mandi/eval_real2code")
    parser.add_argument("--eval_sam_model", default="v4_pointsTrue_lr0.0003_bs24_ac12_02-21_11-45/ckpt_epoch_11", type=str)
    # for global joints 
    parser.add_argument("--global_joints", "-gj", action="store_true")
    args = parser.parse_args()
    
    run_name = os.path.dirname(args.model).split('/')[-1] 
    ckpt_name = os.path.basename(args.model).split('.')[0]
    gen_type = args.gen_type
    ckpt_dir = join(args.eval_data_dir, run_name, ckpt_name, gen_type)
    if args.global_joints:
       print("loading global joint data for this model")
    if args.pool_data:
        pooled = pool_data_across_generations(args)
        exit()
    run(args, ckpt_dir)