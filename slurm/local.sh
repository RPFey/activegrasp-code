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
ws=$(pwd)
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export PKGS_PATH=${ws}/../pkgs/

export LD_LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib
export CUROBO_TORCH_CUDA_GRAPH_RESET=1
export H_TRAIN_POS_ONLY=1
echo $(which python)

for iter in {1..1}
do
    python src/kinova_control/kinova_control_py/curobo_controller.py \
        --ep_root ./grasp_episode \
        --grasp_model ${GRASP_MODEL} \
        --active_view 2 \
        --active_method ${ACTIVE} \
        --episode ${EPISODE_ID} \
        --seed ${SEED} \
        --H_lambda 0.0001 \
        --init_view 2 \
        --data_root ${ws}/../grasp_data_cu/${GRASP_MODEL}_${ACTIVE}_H_lambda1e-4/s${SEED}-ep${EPISODE_ID}/iter${iter}
done