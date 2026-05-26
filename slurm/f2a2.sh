#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=24G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=0:30:00
##SBATCH --exclude=al-l40s-0.grasp.maas
##SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas
##SBATCH --exclude=ee-3090-0.grasp.maas,ee-a6000-0.grasp.maas
##SBATCH --exclude=node-a6000-0,node-v100-0
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/f2a2_new/%x-%A_%a.out
#SBATCH --error=./output/f2a2_new/%x-%A_%a.err
#SBATCH -J apf2a2 

# 2 fixed + 2 active
hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT '/ Job ID' $SLURM_JOBID

SEED=$1
EPISODE_ID=$2
GRASP_MODEL=$3
ACTIVE=$4

echo "SEED $SEED EPISODE_ID $EPISODE_ID"

# source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp

# if hostname has 2080 in it use sepcial conda env
if [[ $(hostname) == *"2080"* ]]; then
    # source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp2080
    source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp2080
    echo "Activating env for 2080ti"
    export PKGS_PATH=/home/leiboshu/ActiveGrasp_2080/pkgs
    export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp2080/lib/libiomp5.so
else
    # source /mnt/kostas-graid/sw/envs/wen/miniforge3/bin/activate /mnt/kostas-graid/sw/envs/wen/miniforge3/envs/regrasp
    source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp
    export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so
    export PKGS_PATH=/home/leiboshu/ActiveGrasp/pkgs/
fi
export LD_LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib
export CUROBO_TORCH_CUDA_GRAPH_RESET=1

ws=~/ActiveGrasp
cd ~/ActiveGrasp/ActiveTouch
export H_TRAIN_POS_ONLY=1

srun python src/kinova_control/kinova_control_py/curobo_controller.py \
    --ep_root ../grasp_episode \
    --grasp_model ${GRASP_MODEL} \
    --active_view 10 \
    --active_method ${ACTIVE} \
    --episode ${EPISODE_ID} \
    --seed ${SEED} \
    --H_lambda 0.0001 \
    --init_view 2 \
    --data_root ${ws}/f2a2_diffsum/${GRASP_MODEL}_${ACTIVE}_score_grasp-bkgd1.5-bullet-s10/s${SEED}-ep${EPISODE_ID}/iter${SLURM_ARRAY_TASK_ID}


