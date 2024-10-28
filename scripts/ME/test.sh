#!/bin/bash

cd ../..

DATA=''


DATASET=pacs
NLAB=105

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

S1=${D2}
S2=${D3}
S3=${D4}
T=${D1}

TRAINER=FBCSA
NET=resnet18

python test.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --source-domains ${S1} ${S2} ${S3} \
    --target-domains ${T} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
    --output-dir output/${DATASET}/nlab_${NLAB}/${TRAINER}/${NET}/${T}/seed${SEED} \
    MODEL.BACKBONE.NAME ${NET} \
    DATASET.NUM_LABELED ${NLAB}