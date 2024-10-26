#!/bin/bash
DATASET=${1:-mvtecad}
DISCOUNT_FACTOR=${2:-0.9}
EPS=${3:-False}
PRETRAIN=${4:-True}
CLASS=${5:-1}

if [ "$DATASET" = "cifar10" ]; then
    MODEL="cifar10_LeNet_ELU"
    LOG_DIR="./log/cifar10_test"
    DATA_DIR="./data"
    SEED=1170014347
    N_EPOCHS=150
    BATCH_SIZE=256
    LR=0.0001
    LR_MILESTONE=120
    WEIGHT_DECAY=0.5e-6
    AE_N_EPOCHS=350
    AE_BATCH_SIZE=256
    AE_LR=0.0001
    AE_LR_MILESTONE=280
    AE_WEIGHT_DECAY=0.5e-6    
elif [ "$DATASET" = "mnist" ]; then
    MODEL="mnist_LeNet"
    LOG_DIR="./log/mnist_test"
    DATA_DIR="./data"
    SEED=-1
    N_EPOCHS=150
    BATCH_SIZE=256
    LR=0.0001
    LR_MILESTONE=120
    WEIGHT_DECAY=0.5e-6
    AE_N_EPOCHS=150
    AE_BATCH_SIZE=256
    AE_LR=0.0001
    AE_LR_MILESTONE=120
    AE_WEIGHT_DECAY=0.5e-3
else
    MODEL="mvtecad_LeNet_ELU"
    LOG_DIR="./log/mvtecad_test"
    DATA_DIR="./data"
    SEED=1758683904
    N_EPOCHS=60
    BATCH_SIZE=32
    LR=0.01
    LR_MILESTONE="20 50"
    WEIGHT_DECAY=0.5e-6
    AE_N_EPOCHS=75
    AE_BATCH_SIZE=32
    AE_LR=0.01
    AE_LR_MILESTONE=60
    AE_WEIGHT_DECAY=0.5e-3
fi

LR_MILESTONE_ARRAY=""
echo "$LR_MILESTONE" | tr ' ' '\n' | while read milestone; do
    LR_MILESTONE_ARRAY="$LR_MILESTONE_ARRAY --lr_milestone $milestone"
done

if [ ! -d "$LOG_DIR/$CLASS" ]; then
    mkdir -p "$LOG_DIR/$CLASS"
fi

nohup python ./src/main.py $DATASET "$MODEL" "$LOG_DIR" "$DATA_DIR" \
    --seed "$SEED" --lr "$LR" --n_epochs "$N_EPOCHS" $LR_MILESTONE_ARRAY \
    --batch_size "$BATCH_SIZE" --weight_decay "$WEIGHT_DECAY" --discount_factor "$DISCOUNT_FACTOR" \
    --eps "$EPS" --pretrain "$PRETRAIN" --ae_lr "$AE_LR" --ae_n_epochs "$AE_N_EPOCHS" \
    --ae_lr_milestone "$AE_LR_MILESTONE" --ae_batch_size "$AE_BATCH_SIZE" \
    --ae_weight_decay "$AE_WEIGHT_DECAY" --normal_class "$CLASS" \
    > "$LOG_DIR/$CLASS/gamma=${DISCOUNT_FACTOR}_eps=${EPS}.out" 2>&1 &