GRASP_MODEL=$1
ACTIVE=$2
for SEED in {0..19}
# for SEED in {0..4}
do
    for EPISODE_ID in {1..5}
    do
        sbatch slurm/f2a2.sh $SEED $EPISODE_ID $GRASP_MODEL $ACTIVE
    done
done