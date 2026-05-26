
# START=$1
# END=$2

for SEED in {103696..103741}
do
    echo "SEED $SEED"
    scancel ${SEED}\_[0-7]
done