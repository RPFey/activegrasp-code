# Active Grasp Project 

## Docker 

```bash
docker run --privileged -it \
        --name dexterous_manipulation_desktop \
        --hostname dexterous_manipulation_desktop \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix \
        -v /path/to/container/mount:/home/user/Documents \
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
git clone git@github.com:RPFey/activegrasp-code.git activegrasp
cd activegrasp

conda env create -f environment.yaml
conda activate grasp

# for server [cluster]
chmod +x ./build.sh 
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" ./build.sh server

# for Desktop
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" ./build.sh all
```

Note, You can specify `TORCH_CUDA_ARCH_LIST="x.x"` according to your GPU arch

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

### Simulator (Pybullet)

```bash
bash slurm/local.sh -s 0 -e 0 --grasp se3diff_dual --active FisherGrasp

```

### Real World (Kinova)

To run the experiment in real world:

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