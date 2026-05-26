# Kinova Setup

For the real kinova experiment.

Use the following command to start image

```bash
docker run -it --runtime=nvidia \
                -e QT_X11_NO_MITSHM=1  \
                --privileged \
                -v /run/udev:/run/udev:ro \
                -e NVIDIA_VISIBLE_DEVICES=all \
                -e NVIDIA_DRIVER_CAPABILITIES=all  \
                --cpus=16 --memory=32g --shm-size=8g \
                -v /home/kostas-lab/Documents/ActiveGrasp_Real:/root \
                -v /dev:/dev \
                --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
                -p 5561:80 -p 5562:5900 -p 5563:22 \
                -e VNC_PASSWORD=rtx4090 -e HTTP_PASSWORD=rtx4090 \
                boshuuu/vnc-cuda:cuda-12.1-devel-ubuntu20.04-gl-ros-noetic
```

I do not recommend use conda env for the ROS. This is because of the libs in conda can corrupt the setup.

Instead, after cloning the repo, inside the repo, run

```bash
./build.sh python
```

to create a virtual python environ in `../grasp` the directory. After activating the python environment, run

```bash
source ../grasp/bin/activate

./build.sh kortex # build ros kortex package for kinova arm
source ../kortex_ws/devel/setup.bash

./build.sh ros # setup ros
./build.sh catkin # build current repo
```