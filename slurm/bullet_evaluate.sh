#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=kostas-compute
#SBATCH --qos=kd-high
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=5:00:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-0.grasp.maas,ee-3090-1.grasp.maas
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/bullet/%x-%A_%a.out

mkdir -p ./output/bullet

hostname
POLICY=$1
# defaults to 8
VIEWS=${2:-8}
source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp
echo "Policy: $POLICY"
echo "Views: $VIEWS"

ws=~/ActiveGrasp

cd ~/ActiveGrasp/ActiveTouch
srun python src/gaussian_splatting/gaussian_splatting_py/grasp/bullet_evaluation.py \
    --num_processes 16 --num_objects 8 --num_seeds 20 --policy $POLICY --num_views $VIEWS 