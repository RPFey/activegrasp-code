# Dev Logs

When thinking about the grasp interface, I come out with two types of methods:

1. Predict the grasp pose and use vel control to approach.

    For these methods, I implement the `predict_grasp_pose` method to give a (N, 4, 4) coords
        for candidate poses in the robot `base_link`.

    The control policy could be:
        1. Move to the pre-grasp pose
        2. Vel contrl to the desired pose.

2. RL related method. For these methods, it will predict the vel command

    For these methods, I implement the `step` method to give a (N, ) velocity vector for the robot.

# Setup

## EquiFlowGrasp

download the pre-trained weight from [here](https://drive.google.com/drive/folders/1H-MXRVcTekdEfzXU_suSw7Afi-7o8I39).

run the scripts

```bash
source devel/setup.bash && export DISPLAY=:1
cd src/kinova_control/kinova_control_py/grasp
python3.11 equiflow_grasp.py --train_result_path /path/to/checkpoint/folder --datadir /path/to/data/folder
```

## Contact Grasp Net

Install Contact Grasp Net from this repo [here](https://github.com/elchun/contact_graspnet_pytorch.git)

run the scripts

```bash
source devel/setup.bash && export DISPLAY=:1
cd src/kinova_control/kinova_control_py/grasp
python3.11 contact_graspnet.py --np_path /path/to/data/folder --ckpt_dir /path/to/contact/checkpoint
```

# Bullet Evaluation 

Go to directory 

```bash
cd src/gaussian_splatting
python gaussian_splatting_py/grasp/bullet_evaluation.py --policy se3diff --num_seeds 16 --num_objects 8 --num_processes 16
```
