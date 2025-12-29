print#!/usr/bin/env bash


set -e

cd src/

###### Prepare experiments for AUMC ######

# Full dataset
python create_dataset_from_custom.py --features_file ../data/custom/Matias/df_long.csv --output_dir ../data/unprocessed/ml_health --pred_window_hours 0
# preprocess full dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health --output_dir ../data/processed/ --dataset_name aumc

# Pred window 4h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/df_long.csv --output_dir ../data/unprocessed/ml_health-4 --pred_window_hours 4
# preprocess -4 dataset 
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-4 --output_dir ../data/processed/ --dataset_name aumc-4

# Pred window 8h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/df_long.csv --output_dir ../data/unprocessed/ml_health-8 --pred_window_hours 8
# preprocess -8 dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-8 --output_dir ../data/processed/ --dataset_name aumc-8

# Pred window 12h
python create_dataset_from_custom.py --features_file ../data/custom/Matias/df_long.csv --output_dir ../data/unprocessed/ml_health-12 --pred_window_hours 12
# preprocess -12 dataset
python preprocess_aumc.py --raw_data_path ../data/unprocessed/ml_health-12 --output_dir ../data/processed/ --dataset_name aumc-12

#### Train models
# Train istrats on full dataset 
python main.py --pretrain 1 --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc/istrats_final
python custom_inference_istrats.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/istrats_final
python custom_inference_istrats.py --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/istrats_final
python custom_inference_istrats.py --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/istrats_final
python custom_inference_istrats.py --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc/istrats_final


# Train istrats model on 4h pred window dataset
python main.py --pretrain 1 --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-4/istrats_final
python custom_inference_istrats.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/istrats_final
python custom_inference_istrats.py --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/istrats_final
python custom_inference_istrats.py --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/istrats_final
python custom_inference_istrats.py --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-4/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-4/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-4/istrats_final

# Train istrats model on 8h pred window dataset
python main.py --pretrain 1 --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-8/istrats_final
python custom_inference_istrats.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/istrats_final
python custom_inference_istrats.py --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/istrats_final
python custom_inference_istrats.py --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/istrats_final
python custom_inference_istrats.py --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-8/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-8/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-8/istrats_final

# Train istrats model on 12h pred window dataset
python main.py --pretrain 1 --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100 --train_batch_size 16
python main.py --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --train_batch_size 16 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --output_dir ../outputs/aumc-12/istrats_final
python custom_inference_istrats.py --dataset aumc --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/istrats_final
python custom_inference_istrats.py --dataset aumc-4 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/istrats_final
python custom_inference_istrats.py --dataset aumc-8 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/istrats_final
python custom_inference_istrats.py --dataset aumc-12 --model_type istrats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ../outputs/aumc-12/pretrain/checkpoint_best.bin --model_ckpt_path ../outputs/aumc-12/istrats_final/checkpoint_best.bin --split infer --batch_size 16 --output_dir ../outputs/aumc-12/istrats_final

