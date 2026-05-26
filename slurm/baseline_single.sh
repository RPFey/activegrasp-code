#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
##SBATCH --array=0-4
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=1:00:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas
##SBATCH --exclude=ee-3090-0.grasp.maas,ee-a6000-0.grasp.maas
##SBATCH --exclude=node-a6000-0,node-v100-0
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/single/%x-%j.out
#SBATCH --error=./output/single/%x-%j.err

hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT '/ Job ID' $SLURM_JOBID

SEED=$1
EPISODE_ID=$2
GRASP_MODEL=$3

echo "SEED $SEED EPISODE_ID $EPISODE_ID"

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp
export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so

ws=~/ActiveGrasp

cd ~/ActiveGrasp/ActiveTouch
srun python src/kinova_control/kinova_control_py/curobo_controller.py \
    --ep_root ../grasp_episode \
    --grasp_model ${GRASP_MODEL} \
    --ep_file ../grasp_episode/seed${SEED}_ep${EPISODE_ID}.json \
    --data_root ${ws}/grasp_data
