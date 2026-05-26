SEED=1
EPISODE_ID=5
GRASP_MODEL=se3diff_ap_band
# ACTIVE=Breyer
ACTIVE=FisherGrasp
# ACTIVE=ActiveNGF
# ACTIVE=ACE

echo "SEED $SEED EPISODE_ID $EPISODE_ID"

# source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp
# conda activate /mnt/kostas-graid/sw/envs/wen/gg2
# export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so
# if hostname has 2080 in it use sepcial conda env
# if [[ $(hostname) == *"2080"* ]]; then
#     source /mnt/kostas-graid/sw/envs/wen/miniforge3/bin/activate /mnt/kostas-graid/sw/envs/wen/grasp2080
#     echo "Activating env for 2080ti"
#     export PKGS_PATH=/home/leiboshu/ActiveGrasp_2080/pkgs
#     export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp2080/lib/libiomp5.so
# else
#     source /mnt/kostas-graid/sw/envs/wen/miniforge3/bin/activate /mnt/kostas-graid/sw/envs/wen/miniforge3/envs/regrasp
#     export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so
#     export PKGS_PATH=/home/leiboshu/ActiveGrasp/pkgs/
# fi
source /mnt/kostas-graid/sw/envs/wen/miniforge3/bin/activate /mnt/kostas-graid/sw/envs/wen/grasp
echo "Activating env for 2080ti"
export LD_PRELOAD=/mnt/kostas-graid/sw/envs/boshu/miniconda3/envs/grasp/lib/libiomp5.so
export PKGS_PATH=/home/leiboshu/ActiveGrasp/pkgs/

export LD_LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib
export CUROBO_TORCH_CUDA_GRAPH_RESET=1

ws=~/ActiveGrasp

cd ~/ActiveGrasp/ActiveTouch

echo $(which python)

python src/kinova_control/kinova_control_py/curobo_controller.py \
    --ep_root ../grasp_episode \
    --grasp_model ${GRASP_MODEL} \
    --active_method ${ACTIVE} \
    --episode ${EPISODE_ID} \
    --active_view 1 \
    --init_view 1 \
    --seed ${SEED} \
    --data_root ${ws}/conformal_data_cu/${GRASP_MODEL}_${ACTIVE}/s${SEED}-ep${EPISODE_ID}/iter1