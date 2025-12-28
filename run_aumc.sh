#!/bin/bash

# run_commands(){
#    eval template="$1"
#    for train_frac in 0.5 0.4 0.3 0.2 0.1; do
#        for ((i=1; i<=10; i++)); do
#            run_param="${i}o10"
#            eval "$1 --run $run_param --train_frac $train_frac"
#        done
#    done
#}

cd src/

###### Prepare experiments for AUMC ######

# Full dataset
python create_dataset_from_custom.py --features_file ../data/custom/Matias/output_1.1.csv --output_dir ../data/unprocessed/ml_health --pred_window_hours 0

# preprocess full dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health --output_dir ../data/processed/ --dataset_name aumc

# Pred window 4h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/output_1.1.csv --output_dir ../data/unprocessed/ml_health-4 --pred_window_hours 4
# preprocess -4 dataset 
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-4 --output_dir ../data/processed/ --dataset_name aumc-4

# Pred window 8h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/output_1.1.csv --output_dir ../data/unprocessed/ml_health-8 --pred_window_hours 8
# preprocess -8 dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-8 --output_dir ../data/processed/ --dataset_name aumc-8
# todo test

# Pred window 12h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/output_1.1.csv --output_dir ../data/unprocessed/ml_health-12 --pred_window_hours 12
# preprocess -12 dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-12 --output_dir ../data/processed/ --dataset_name aumc-12

#### Train models
# Train Strats on full dataset 
python main.py --pretrain 1 --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc/strats_final
python custom_inference.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/strats_final
python custom_inference.py --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/strats_final
python custom_inference.py --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/strats_final
python custom_inference.py --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/strats_final


# Train Strats model on 4h pred window dataset
python main.py --pretrain 1 --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-4/strats_final
python custom_inference.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/strats_final
python custom_inference.py --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/strats_final
python custom_inference.py --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/strats_final
python custom_inference.py --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/strats_final

# Train Strats model on 8h pred window dataset
python main.py --pretrain 1 --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-8/strats_final
python custom_inference.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/strats_final
python custom_inference.py --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/strats_final
python custom_inference.py --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/strats_final
python custom_inference.py --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/strats_final

# Train Strats model on 12h pred window dataset
python main.py --pretrain 1 --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-12/strats_final
python custom_inference.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/strats_final
python custom_inference.py --dataset aumc-4 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/strats_final
python custom_inference.py --dataset aumc-8 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/strats_final
python custom_inference.py --dataset aumc-12 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/strats_final




### python main.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc/strats_final 
## Evaluation or inference
# python .\custom_inference.py --dataset aumc --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/strats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/strats_final


# python main.py --pretrain 1 --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 8
# template="python main.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 8 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin"
# run_commands "\${template}"