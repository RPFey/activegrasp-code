# Active Grasp Project 

## Docker 

```bash
docker run --privileged -it \
        --name dexterous_manipulation_desktop \
        --hostname dexterous_manipulation_desktop \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix \
        -v /home/wen/Documents/boshu_docker_ws:/home/user/Documents \
        --cpus=32 --memory=64g --shm-size=16g \
        --device=/dev/dri:/dev/dri \
        -p 4582:22 \
        --device=/dev:/dev \
        --env="DISPLAY=$DISPLAY" \
        -e "TERM=xterm-256color" \
        --cap-add SYS_ADMIN --device /dev/fuse \
        --gpus all \
        peasant98/dexterous_manipulation_desktop:latest \
        bash
```

Quick Setup for `diff-eig`, `FoundationPose`, `Contact Grasp Net`, `ROS`, `Realsense`, `gsplat`.

```bash
mkdir -p ws/ActiveGrasp
cd ws/ActiveGrasp
git clone git@github.com:RPFey/ActiveTouch.git
cd ActiveTouch

conda env create -f environment.yaml
conda activate grasp

# for server [cluster]
chmod +x ./build.sh 
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" ./build.sh server

# run 
sbatch slurm/baseline.sh 4 # seed 4 scene

# for Desktop
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" ./build.sh all
```

Note, You can specify `TORCH_CUDA_ARCH_LIST="x.x"` according to your GPU arch

**For diff_eig, make sure your machine has the correct key!**

## System Design

Package 

1. kinova control package

2. vision package

I plan to separate gaussian splatting model from ROS so that it is better for debugging and tuning.

Python Package (independent of ROS)

--- 

Packages inside this repo

* Gaussian Splatting (Provides API for vision package)

* Grasp (Grasp generation)

* FisherRF package

* DepthAnyThing

--- 

Packages outside this repo

* SAM2 

* Open3D (Provides SDF API)

* CuRobo (replace KDL Lib in the future)

## Setup

## Pybullet Simulation

```bash
# clone repo 
# it does not need to be under the current repo
git clone https://github.com/RPFey/franka_pybullet_ros.git
python -m pip install pybullet

## ROS
# roslaunch kinova_control curobo_pybullet.launch
source .bashrc 
source devel_isolated/setup.bash && export DISPLAY=:1
GS_DATA=/root/data/2025-01-21-21-06-00 RUN_CONFORMAL=1 roslaunch kinova_control curobo_pybullet.launch 
# In terminal 2, under franka_pybullet_ros folder run 
source .bashrc && export DISPLAY=:1
python ros_example_physics.py

## w.o ROS
cd src/kinova_control/kinova_control_py
python curobo_controller.py --ep_file /path/to/episode_json --data_root /path/to/save_data
```

## Note when running conformal prediction:

The following temp files need to be removed:
```
/tmp/grasp_poses_scores.npz
/tmp/done.grasp_record
/tmp/done.grasp_record
```

Running in batch:
First we need to run pybullet simulation to get `gs_data` and `grasp_poses` saved, then:

```bash
GS_DATA=/root/data/2025-01-21-21-06-00 RUN_CONFORMAL=1 python run_conformal.py 
```

## Set up FoundationPose
```
# install dependencies
cd src/gaussian_splatting
python3.11 -m pip install -r requirements_fp.txt
python3.11 -m pip install --upgrade psutil

# Install NVDiffRast
python3.11 -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python3.11 -m pip install --quiet --no-cache-dir kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

python3.11 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build extensions
cd ../gaussian_splatting_py/FoundationPose
CMAKE_PREFIX_PATH=/home/user/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 bash build_all_docker.sh
```

Get weights
```
python3.11 -m pip install gdown
mkdir weights; cd weights
gdown --folder --id 1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC
gdown --folder --id 12Te_3TELLes5cim1d7F7EBTwUSe7iRBj
gdown --folder --id 1LUyJpzvR_UicpLDrTHlQgU3x06akPeZ_
gdown --id 1_C1cRwPRbSn2zuV20F36XPGF77OKgugv
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true -O depth_anything_v2_vitl.pth
cd ..
```

If you want to run the demo to check it is working
```
mkdir demo_data; cd demo_data
gdown https://drive.google.com/uc?id=1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3
unzip mustard0.zip; rm mustard0.zip
cd ..
python3.11 run_demo.py  --debug 0
```

The STL File can be found here, [link](https://github.com/hsp-iit/GRASPA-benchmark/tree/master/data/objects/YCB)

## Set up GroundedSAM2
```
cd src/gaussian_splatting/gaussian_splatting_py
git submodule update --init Grounded-SAM-2/
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12.1/
python3.11 -m pip install -e .
python3.11 -m pip install --no-build-isolation -e grounding_dino
python3.11 -m pip install supervision #required for debug visualizations
python3.11 -m pip install transformers addict yapf pycocotools timm
```

Get weights
```
cd gdino_checkpoints
bash download_ckpts.sh
cd ../checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Build Issues

* PyKDL

add following lines in the `orocos_kinematics_dynamics/python_orocos_kdl` CMakeLists.txt

```
# insert this between line 17-18
include_directories(pybind11/include)

# insert this between line 26-27
SET(PYTHON_VERSION 3.11)
```

* DeadLock Issue

It seems that Python3.11 will stuck at rospy.init_node, please run 

```bash

sudo cp dump/roslogging.py /opt/ros/noetic/lib/python3/dist-packages/rosgraph/roslogging.py
```

* Gsplat Issue

Gsplat requires compute capability >= 7.0. If you encounter compile problems and you are sure your GPU satisfies the requirement, you can specify it during compilation.

```bash
# V100 is 700
TORCH_CUDA_ARCH_LIST="7.0" python3.11 -m pip install . -v
# for RTX4090
TORCH_CUDA_ARCH_LIST="9.0" python3.11 -m pip install . -v
```

## RUN

```bash
# terminal 1
source devel/setup.bash && export DISPLAY=:1
roslaunch kinova_control ros_kortex.launch

# terminal 2
source devel/setup.bash && export DISPLAY=:1
export PYTHONPATH=/home/user/Documents/librealsense/build/Release:$PYTHONPATH
roslaunch kinova_control moveit_controller.launch
```


Misc:

Coordinate sysmte of the robot arm: `x: forward` `y: left` `z: up`

```
x: forward
         z
         ^   ^ x
         |  /
         | /
y <______|/
```

### Running viewers to check collected data:

```bash
cd src/gaussian_splatting
python gaussian_splatting_py/cam_viewer.py ~/data/first_touch --share
```

### Running Proto-Comp

```bash
export PYTHONPATH=$PYTHONPATH:/home/user/Documents/pointnet2_ops
```

packages required:

```bash
git clone https://github.com/fishbotics/pointnet2_ops
cd pointnet2_ops
python setup.py install # NOTE: pip install might install cpu version of torch because they only check torch>=1.8 but failed on 1.2+cu etc
```

```bash
python3.11 -m pip install git+https://github.com/openai/CLIP.git
cd extensions/chamfer_dist && python3.11 -m pip install .
python3.11 -m pip install transforms3d tensorboardX ipdb
```

Under Proto-comp-pip repo:
```bash
python3.11 -m pip install -e . -v
```

### debug splatting componets:

```bash
python splatting.py --run_dir /root/ActiveGrasp/conformal_data_cu/se3diff_FisherGrasp/s2-ep3/iter1/2025-04-17-20-19-01-s2-ep3 --result_dir /tmp/debug-hess/ --load_ckpt  /root/ActiveGrasp/conformal_data_cu/se3diff_FisherGrasp/s2-ep3/iter1/2025-04-17-20-19-01-s2-ep3/gs/ckpts/ckpt_800_rank0.pt
```

```
export PATH=/mnt/kostas-graid/sw/envs/wen/miniforge3/envs/regrasp/bin:$PATH
CC=/usr/bin/gcc CXX=/usr/bin/g++ python3 -m pip install -e . --no-build-isolation


export TORCH_CUDA_ARCH="7.5,8.0;8.6;8.9"
```