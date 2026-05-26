#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=24G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=0:20:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-0.grasp.maas,ee-3090-1.grasp.maas,al-l40s-0.grasp.maas
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/f2a2_new/%x-%A_%a.out
#SBATCH --error=./output/f2a2_new/%x-%A_%a.err
#SBATCH -J apf2a2_lambda

# 4 fixed + 2 active
hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT '/ Job ID' $SLURM_JOBID

SEED=$1
EPISODE_ID=$2
GRASP_MODEL=$3
ACTIVE=$4
ITER=$5

echo "SEED $SEED EPISODE_ID $EPISODE_ID"

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp
export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so
export PKGS_PATH=/home/leiboshu/ActiveGrasp/pkgs/
export LD_LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib
export CUROBO_TORCH_CUDA_GRAPH_RESET=1

ws=~/ActiveGrasp
cd ~/ActiveGrasp/ActiveTouch
export H_TRAIN_POS_ONLY=1

srun python src/kinova_control/kinova_control_py/curobo_controller.py \
    --ep_root ../grasp_episode \
    --grasp_model ${GRASP_MODEL} \
    --active_view 2 \
    --active_method ${ACTIVE} \
    --episode ${EPISODE_ID} \
    --seed ${SEED} \
    --H_lambda 0.0001 \
    --init_view 2 \
    --data_root ${ws}/f2a2-grasp/${GRASP_MODEL}_${ACTIVE}_score_grasp-bkgd1.5-1e4-g512/s${SEED}-ep${EPISODE_ID}/iter${ITER}


