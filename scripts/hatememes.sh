DATASET="hatememes"
for MISSING_RATIO in 0.5 0.7 0.9
do
    for MISSING_TYPE in both text image
    do
        python src/main.py experiment=mora_hatememes dataset=${DATASET} dataset.missing_params.RATIO=${MISSING_RATIO} dataset.missing_params.TYPE=${MISSING_TYPE} EXP_NOTE="MoRA_WORec_${DATASET}_${MISSING_RATIO}_${MISSING_TYPE}" train.EARLY_STOPPING=20
    done
done