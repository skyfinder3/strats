#!/bin/bash

run_commands(){
    eval template="$1"
    for train_frac in 0.5 0.4 0.3 0.2 0.1; do
        for ((i=1; i<=10; i++)); do
            run_param="${i}o10"
            eval "$1 --run $run_param --train_frac $train_frac"
        done
    done
}

cd src/

# get data form data/custom aka. cohort and values into shape
python create_dataset_from_custom.py

# preprocess dataset like physionet12
python preprocess_aumc.py
# Strats custom with pretrain for 100 epochs
python main.py --pretrain 1 --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100
template="python main.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin"
run_commands "\${template}"

python main.py --pretrain 1 --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100
template="python main.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin"
run_commands "\${template}"