#!/bin/bash

# Default values
SEED=0
EPISODE_ID=10
GRASP_MODEL=se3diff_scene
ACTIVE=FisherGrasp

while [[ $# -gt 0 ]]; do
  case $1 in
    -s)
      SEED="$2"
      shift 2
      ;;
    -e)
      EPISODE_ID="$2"
      shift 2
      ;;
    --grasp)
      GRASP_MODEL="$2"
      shift 2
      ;;
    --active)
      ACTIVE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -s <start> -e <end> --grasp <type> [--active]"
      exit 1
      ;;
  esac
done

# Print results
echo "Seed    = $SEED"
echo "Episode = $EPISODE_ID"
echo "Grasp   = $GRASP_MODEL"
echo "Active  = $ACTIVE"

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
export H_TRAIN_POS_ONLY=1

ws=~/ActiveGrasp
cd ~/ActiveGrasp/ActiveTouch
echo $(which python)

for iter in {1..1}
do
    python src/kinova_control/kinova_control_py/curobo_controller.py \
        --ep_root ../grasp_episode \
        --grasp_model ${GRASP_MODEL} \
        --active_view 1 \
        --active_method ${ACTIVE} \
        --episode ${EPISODE_ID} \
        --seed ${SEED} \
        --H_lambda 0.001 \
        --init_view 1 \
        --data_root ${ws}/conformal_data_cu/${GRASP_MODEL}_${ACTIVE}_H_lambda1e-3/s${SEED}-ep${EPISODE_ID}/iter${iter}
done