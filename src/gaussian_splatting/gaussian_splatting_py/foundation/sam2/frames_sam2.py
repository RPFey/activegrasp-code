"""Python3.11* code to segment images in a folder using a UI with Segment Anything 2."""

import argparse
import json
import json as js
import os
import os.path as osp
import pickle
import shutil
import tkinter as tk
from tkinter import Canvas
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from gaussian_splatting_py.foundation.grounding_dino_wrapper import \
    GroundingDINO
from PIL import Image, ImageTk
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

# Disable this, since Foundation Pose needs float32, this is moved to decorator for SAM2
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

parent_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..")
SAM2_PATH = osp.join(parent_folder, 'weights', 'sam2.1_hiera_large.pt') #
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_POINTS_FILENAME = 'points_labels.json'

# SAM2 UI on multiple frames (video or not)

def save_mask(mask, root_dir, out_path):
    """Save the mask as a PNG image."""
    mask_root = os.path.join(root_dir, 'masks')
    os.makedirs(mask_root, exist_ok=True)
    mask_path = os.path.join(mask_root, out_path)
    mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    mask = mask.astype(np.uint8) * 255
    mask_image = Image.fromarray(mask)
    mask_image.save(mask_path)

def show_mask(mask, ax, obj_id=None, random_color=False):
    """Show the mask on the matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, canvas, point_radius=5):
    """Show the points on the canvas."""
    for coord, label in zip(coords, labels):
        x, y = coord
        color = 'green' if label == 1 else 'red'
        canvas.create_oval(x - point_radius, y - point_radius, x + point_radius, y + point_radius, fill=color, outline='white')

def on_click(event, points, labels, point_type, canvas):
    """Add a point on the canvas."""
    x, y = event.x, event.y
    points.append([x, y])
    labels.append(point_type[0])
    show_points(np.array([[x, y]]), np.array([point_type[0]]), canvas)

def on_enter(event, inference_state, predictor, ann_frame_idx, ann_obj_id, frame_names, video_dir, points, labels):
    """Propagate the masks and save the points and labels."""
    
    points_np = np.array(points, dtype=np.float32)
    labels_np = np.array(labels, np.int32)
    
    # save points and labels as pkl file
    root_dir = video_dir.split('/')[:-1]
    exp_name = root_dir[-1]
    seed_ep = exp_name.split('-')[-1].split('_')[0]
    seed = int(seed_ep[4:])
    root_dir = f'/{os.path.join(*root_dir)}'
    
    points_labels = {'points': points_np, 'labels': labels_np}
    # points_labels_path = os.path.join(root_dir, SAM2_POINTS_FILENAME)
    # with open(points_labels_path, 'wb') as f:
    #     pickle.dump(points_labels, f)
        
    points_label = { "points": [p.astype(np.int32).tolist() for p in points_np], "labels": labels_np.tolist() }
    data = {"prompt": points_label, "seed": seed}
    with open(os.path.join(root_dir, SAM2_POINTS_FILENAME), "w") as f:
        json.dump(data, f, indent=4)
        
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_np,
        labels=labels_np,
        box=None,
    )    
    
    compute_masks(inference_state, predictor, ann_frame_idx, 
                  ann_obj_id, labels_np, root_dir, 
                  video_dir, frame_names)
        
    return


def compute_masks(inference_state, predictor, 
                  ann_frame_idx, ann_obj_id, 
                  labels_np, root_dir,
                  video_dir, frame_names, skip_first=False):
    """Compute the masks for the video/multiple frames."""

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, 0):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        # logits = out_mask_logits[0, 0].cpu().numpy()
        # plt.figure()
        # plt.imshow(logits)
        # plt.savefig(f"mask_{out_frame_idx}.png")

    with open(os.path.join(root_dir, "transforms.json"), "r") as f:
        stats = json.load(f)

    frames_idx = [idx for idx, f in enumerate(stats["frames"]) if "rs_image_path" in f]
    if len(frames_idx) == 0:
        frames_idx = [idx for idx, f in enumerate(stats["frames"]) if "file_path" in f]
    
    # sort based on index
    frames_idx.sort(key=lambda x: int(os.path.splitext(stats["frames"][x]["file_path"].split('/')[-1])[0]))

    vis_frame_stride = 1
    valid_frames_num = len(frame_names) 
    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride), desc="Visualizing masks"):
        
        out_mask = video_segments[out_frame_idx][1]
        
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #     show_mask(out_mask, ax, obj_id=out_obj_id)
        # plt.show()
        
        frame_name = os.path.splitext(frame_names[out_frame_idx])[0]
        save_mask(out_mask, root_dir, f"{frame_name}.png")
        stats["frames"][frames_idx[out_frame_idx]]["mask"] = f"masks/{frame_name}.png"
    
    # remove the orinal transforms.json file
    os.remove(os.path.join(root_dir, "transforms.json"))
    with open(os.path.join(root_dir, "transforms.json"), "w") as f:
        json.dump(stats, f, indent=4)

def start_app(data_dir, video_dir, predictor, 
                category: Optional[str]=None, initial_frame:str=None):
    """
    categroy: str, should be in form like "car. tire."
    initial_frame: str, the reference frame prompt
    """
    frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    # sort images based on their index 
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    if initial_frame is not None:
        img = Image.open(initial_frame)
        img = img.convert('RGB')
        img.save(os.path.join(video_dir, "{:04d}.jpg".format(len(frame_names))), 'JPEG')
    
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0 if initial_frame is None else len(frame_names)
    ann_obj_id = 1
    
    points_np = None
    labels_np = None
    bbox_np = None

    # see if points exists 
    root_dir = video_dir.split('/')[:-1]
    root_dir = f'/{os.path.join(*root_dir)}'

    if category is not None:
        if not category.endswith('.'):
            category += '.'
        
        bbox_np = None
        # run grounding dino to get masks
        grounding_dino = GroundingDINO()
        for i in tqdm(range(len(frame_names)), desc="Grounding DINO"):
            image = Image.open(os.path.join(video_dir, frame_names[i]))
            results = grounding_dino(image, category)
            boxes = results[0]["boxes"]
            if len(boxes) > 0 and results[0]["scores"].max() > 0.5: 
                print(f"Found {len(boxes)} boxes in frame {i}")
                bbox_np = boxes.cpu().numpy()[0:1]
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=ann_obj_id,
                    points=None,
                    labels=None,
                    box=bbox_np,
                )

    else:
        points_labels_path = os.path.join(root_dir, SAM2_POINTS_FILENAME)
        if os.path.exists(points_labels_path):
            with open(points_labels_path, 'rb') as f:
                points_obj = js.load(f)
            
            points_np = np.array(points_obj["prompt"]['points'])
            labels_np = np.array(points_obj["prompt"]['labels'])
            print("No need to annotate, using existing points..")

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels_np,
                box=None,
            ) 
    
    if points_np is not None or bbox_np is not None:
        compute_masks(inference_state, predictor, ann_frame_idx, 
                      ann_obj_id, labels_np, root_dir, 
                      video_dir, frame_names, skip_first=initial_frame is not None)
        
        # remove the video dir
        shutil.rmtree(video_dir)
        
        return

    # Points and labels storage
    points = []
    labels = []

    # Use a list to store point type (1 for positive, 0 for negative)
    point_type = [1]

    # Create the main window
    root = tk.Tk()
    root.title("Image Click UI")

    # Load the first image
    img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    img_tk = ImageTk.PhotoImage(img)

    # Create a canvas and display the image
    canvas = Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Bind the click event to add points
    canvas.bind("<Button-1>", lambda event: on_click(event, points, labels, point_type, canvas))

    # Bind the 'p' key to switch to adding positive points
    root.bind('p', lambda event: point_type.__setitem__(0, 1))

    # Bind the 'n' key to switch to adding negative points
    root.bind('n', lambda event: point_type.__setitem__(0, 0))

    # Bind the Enter key to propagate masks
    root.bind('<Return>', lambda event: on_enter(event, inference_state, predictor, ann_frame_idx, ann_obj_id, frame_names, video_dir, points, labels))
    
    
    print("Starting UI...")
    # Start the Tkinter main loop
    root.mainloop()


def convert_images_to_jpg(data_dir):
    """Goes through an `image_dir` and converts all images to JPG format."""
    
    jpg_dir = os.path.join(data_dir, 'jpg_images')
    os.makedirs(jpg_dir, exist_ok=True)
    
    img_dir = os.path.join(data_dir, 'rs_images')
    images = os.listdir(img_dir)
    if len(images) == 0:
        # switch to default
        img_dir = os.path.join(data_dir, 'images')
        images = os.listdir(img_dir)

    for filename in images:
        file_path = os.path.join(img_dir, filename)
        if 'depth' in filename:
            continue
        
        base, ext = os.path.splitext(filename)
        if ext.lower() not in ['.jpg', '.jpeg']:
            img = Image.open(file_path)
            img = img.convert('RGB')
            img.save(os.path.join(jpg_dir, f"{base}.jpg"), 'JPEG')
        else:
            img = Image.open(file_path)
            img.save(os.path.join(jpg_dir, filename), 'JPEG')
    
    return jpg_dir

@torch.amp.autocast('cuda', dtype=torch.bfloat16)
def run_video_prediction(data_dir, category: Optional[str] = None, initial_frame:str=None):
    predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_PATH)
    # Convert images to jpg and store them in 'jpg_images' directory
    jpg_dir = convert_images_to_jpg(data_dir)
    # Run the application with the converted jpg images
    start_app(data_dir, jpg_dir, predictor, category=category, initial_frame=initial_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory in nerfstudio format and convert them to JPG for UI segmentation.")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    parser.add_argument("--category", type=str, help="categroy of the object of interest", default=None)
    
    args = parser.parse_args()
    run_video_prediction(args.data_dir, category=args.category)
