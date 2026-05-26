# Repo for Improved 2D and 3D Gaussian Splatting

## Setup

### Depth Anything V2 Setup

The scripts of Depth Anything V2 is stored in this repo. You only need to download the weight and put them under the `weights` folder. 

```text
-- gaussian_splatting
    |
    -- weights
        |
        -- depth_anything_v2_vitl.pth
```

Also, please check that the encoder type of depth anything v2 matches the weight type, in the `create_model` function in `gaussian_splatting_py/depth_anything_v2/__init__.py`. The encoder type and model weight path is specified in the vision node launch file.

### SAM2 Setup

The SAM2 repo has already been built in this Docker Container. 

### Romatch Setup

Install using the official repo [here](https://github.com/Parskatt/RoMa).

### Digit Setup

Digit is incorporated into the current repo. To run the digit sensor offline, 

```bash
cd src/gaussian_splatting && export PYTHONPATH=$(pwd):$PYTHONPATH
cd gaussian_splatting_py

# to visualize 
python3.11 digit/run_digit.py --task capture --config ../config/digit.yaml --model /path/to/model --background /path/to/background 

# to save captures
python3.11 digit/run_digit.py --task capture --config ../config/digit.yaml --save_data --save_dir /path/to/save --model /path/to/model --background /path/to/background 

# to visualize point cloud
python3.11 digit/run_digit.py --task point_cloud  --config ../config/digit.yaml --img /path/to/digit_image --model /path/to/model --background /path/to/background 
```

## Data Format

All data is saved chronologically. 

### Depth

# Run 

Run offline experiments

```bash
cd src/gaussian_splatting
export PYTHONPATH=$(pwd):$PYTHONPATH
python3.11 gaussian_splatting_py/splatting.py default --init_type romatch --data_dir /home/user/Documents/data/2024-12-19-19-49-45 --max_steps 8000 --no-depth_loss --result_dir ./results/cup
```

## Running uncertainty visualization:

example command:
```bash
cd src/gaussian_splatting
export PYTHONPATH=$(pwd):$PYTHONPATH
python gaussian_splatting_py/uncern_viewer.py --ckpt ./results/num20-s1k/ckpts/ckpt_999_rank0.pt --data_dir ~/data/2024-10-10-00-25-29/
```

## Utility Script

1. Convert our data format to colmap. 

I implement convert to colmap method for the Kinova Dataset. 

2. Visualize and Debug Foundation Model results.

```bash
# visualize monocular depth estimation
python3.11 foundation.py --img_path /path/to/color1 --real_depth /path/to/depth1 --task depth

# visualize SAM2 mask segmentation
python3.11 foundation.py --img_path /path/to/color1 --real_depth /path/to/depth1 --task sam2

# visualize SAM Mask depth alignment
python3.11 foundation.py --img_path /path/to/color1 --real_depth /path/to/depth1 --task depth_mask
```

3. Visualize Depth Alignment results.

This script will create one point cloud from the first 3 rgbd(aligned mono depth) readings and transform them to world coordinate. It will show these point clouds to check the depth alignment results.

```bash
python3.11 tools/depth_check.py --dataset /path/to/dataset
```

4. Cam 2 Cam calibration

To calibrate the realsense to the built in camera, take multiple images of the calibration board from realsense and built in camera at different poses and store them in two separate folders `/path/to/kinova` and `/path/to/realsense`.The board looks like this ![image](asset/0006.png)

You should contain the word **kinova** in the path to images taken from kinova and the word **realsense** in the path to images taken from realsense.

Then run the script

```bash
python3.11 cam2cam.py --imgs1 /path/to/kinova --imgs2 /path/to/realsense
```

It will print the transformation that transform points from the second camera to the first camera.

You can also add `--default_intrinsic` to use the default intrinsic parameters of the cameras instead of using checkerboard. 
