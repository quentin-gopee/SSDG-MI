#!/bin/bash

cd ../..

DATA='~/data'

GAMMA=10
BATCH_SIZE=48
BASELINE=fixmatch

DATASET=$1
NLAB=$2 # total number of labels

if [ ${DATASET} == ssdg_pacs ]; then
    # NLAB: 210 or 105
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_officehome ]; then
    # NLAB: 1950 or 975
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == ssdg_digits_dg ]; then
    # NLAB: 300 or 150
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
elif [ ${DATASET} == ssdg_vlcs ]; then
    # NLAB: 150 or 75
    D1=caltech
    D2=labelme
    D3=pascal
    D4=sun
fi

TRAINER=ME
NET=resnet18

# File to store configs that need training
NEED_TRAINING_FILE=scripts/FBCSA/configs/ME_${DATASET}.json
# Initialize the JSON file
echo "[]" > ${NEED_TRAINING_FILE}

# Define your available GPU IDs
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]} # Number of available GPUs
echo "Number of available GPUs: ${NUM_GPUS}"

# Keep track of the GPU index
GPU_INDEX=0

set_domains() {
    local SETUP=$1
    if [ ${SETUP} == 1 ]; then
        S1=${D2}
        S2=${D3}
        S3=${D4}
        T=${D1}
    elif [ ${SETUP} == 2 ]; then
        S1=${D1}
        S2=${D3}
        S3=${D4}
        T=${D2}
    elif [ ${SETUP} == 3 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D4}
        T=${D3}
    elif [ ${SETUP} == 4 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D3}
        T=${D4}
    fi
}

# Function to check directories and update configs that need training
check_configs() {
    for IMBALANCE in exp_l_only #original
    do
        for SETUP in 1 2 3 4
        do
            for SEED in 1 2 3 4 5
            do
                for ALPHA in 0 #1 2 3 5 7
                do
                    set_domains ${SETUP}

                    if [ ${ALPHA} == 0 ]; then
                        DIRECTORY=output/${DATASET}/nlab_${NLAB}/${BASELINE}/${IMBALANCE}/ME/baseline_d/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED}
                    else
                        DIRECTORY=output/${DATASET}/nlab_${NLAB}/${BASELINE}/${IMBALANCE}/ME/alpha_${ALPHA}/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED}
                    fi

                    if [ -d "$DIRECTORY" ]; then
                        if [ -d "$DIRECTORY/C" ] && [ -d "$DIRECTORY/G" ]; then
                            echo "Config already trained: $IMBALANCE $BATCH_SIZE $LAMBDA $SETUP $SEED $ALPHA"
                        else
                            if [ -e "$DIRECTORY/log.txt" ]; then
                                mv $DIRECTORY/log.txt $DIRECTORY/log.txt-old-$(date +%Y%m%d%H%M%S)
                            fi
                            if [ -d "$DIRECTORY/tensorboard" ]; then
                                rm -r $DIRECTORY/tensorboard
                            fi
                            jq ". += [{\"imbalance\": \"${IMBALANCE}\", \"setup\": ${SETUP}, \"seed\": ${SEED}, \"alpha\": ${ALPHA}}]" ${NEED_TRAINING_FILE} > tmp.json && mv tmp.json ${NEED_TRAINING_FILE}
                        fi
                    else
                        jq ". += [{\"imbalance\": \"${IMBALANCE}\", \"setup\": ${SETUP}, \"seed\": ${SEED}, \"alpha\": ${ALPHA}}]" ${NEED_TRAINING_FILE} > tmp.json && mv tmp.json ${NEED_TRAINING_FILE}
                    fi
                done
            done
        done

    done
}

# # Main training loop
while true; do
    # Reset the need_training.json file
    echo "[]" > ${NEED_TRAINING_FILE}

    # Check directories for configs that need training
    check_configs

    # Get the number of configs that need training
    NUM_TRAINING=$(jq length ${NEED_TRAINING_FILE})
    echo "Number of configs that need training: ${NUM_TRAINING}"

    if [ ${NUM_TRAINING} -eq 0 ]; then
        echo "All configs are trained!"
        break
    fi

    echo "Training remaining configs..."
    for row in $(jq -c '.[]' ${NEED_TRAINING_FILE}); do
        IMBALANCE=$(echo $row | jq -r '.imbalance')
        SETUP=$(echo $row | jq -r '.setup')
        SEED=$(echo $row | jq -r '.seed')
        ALPHA=$(echo $row | jq -r '.alpha')

        set_domains ${SETUP}

        echo "Training config: imbalance=${IMBALANCE}, setup=${SETUP}, seed=${SEED}, alpha=${ALPHA}"

        # Wait for a GPU to become available before launching a new experiment
        echo "Waiting for a GPU to become available..."
        while [ $(jobs -r | wc -l) -ge ${NUM_GPUS} ]; do
            sleep 1
        done

        if [ ${ALPHA} == 0 ]; then
            DIRECTORY=output/${DATASET}/nlab_${NLAB}/${BASELINE}/${IMBALANCE}/ME/baseline_d/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED}
            # Assign a GPU and run the process in the background
            CUDA_VISIBLE_DEVICES=${GPUS[GPU_INDEX]} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --source-domains ${S1} ${S2} ${S3} \
            --target-domains ${T} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
            --output-dir ${DIRECTORY} \
            --imbalance ${IMBALANCE} \
            --gamma ${GAMMA} \
            --batch-size ${BATCH_SIZE} \
            --baseline ${BASELINE} \
            MODEL.BACKBONE.NAME ${NET} \
            DATASET.NUM_LABELED ${NLAB} &
        else
            DIRECTORY=output/${DATASET}/nlab_${NLAB}/${BASELINE}/${IMBALANCE}/ME/alpha_${ALPHA}/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED}
            # Assign a GPU and run the process in the background
            CUDA_VISIBLE_DEVICES=${GPUS[GPU_INDEX]} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --source-domains ${S1} ${S2} ${S3} \
            --target-domains ${T} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
            --output-dir ${DIRECTORY} \
            --imbalance ${IMBALANCE} \
            --gamma ${GAMMA} \
            --me alpha \
            --weight-h ${ALPHA} \
            --batch-size ${BATCH_SIZE} \
            --baseline ${BASELINE} \
            MODEL.BACKBONE.NAME ${NET} \
            DATASET.NUM_LABELED ${NLAB} &
        fi

        # Move to the next GPU
        GPU_INDEX=$(( (GPU_INDEX + 1) % ${NUM_GPUS} ))
    done
    # Wait for all background processes to finish before exiting
    wait
    echo "Rechecking for remaining configs..."
done