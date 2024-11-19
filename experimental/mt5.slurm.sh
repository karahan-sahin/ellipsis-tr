#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --job-name=ellipsis-mt5-release
#SBATCH -o ../ellipsis-mt5.out
#SBATCH -t 2-0:00:00

source /opt/python3/venv/base/bin/activate

cd /users/karahan.sahin/ellipsis-tr
ls -ah
pip install -r requirements.txt

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.release.train.csv' \
                                                --model_name="google/mt5-base" \
                                                --output_dir='models/ellipsis-type-mt5-release' \
                                                --hub_model_id='ellipsis-type-mt5-release' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=64 \
                                                --per_device_eval_batch_size=8 \
                                                --gradient_accumulation_steps=8 \
                                                --learning_rate=5e-5 \
                                                --eval_steps=50 \
                                                --model_type='encoder' \
                                                --min_count_per_class=70 \
                                                --max_count_per_class=5000 \
                                                --over_sample \
                                                --num_epochs=50

rm -rf models/*


python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.release.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --output_dir='models/ellipsis-discriminative-mt5-release' \
                                                 --hub_model_id='ellipsis-discriminative-mt5-release' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='discriminative' \
                                                 --learning_rate=5e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50

rm -rf models/*

python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.release.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --output_dir='models/ellipsis-extractive-mt5-release' \
                                                 --hub_model_id='ellipsis-extractive-mt5-release' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='extractive' \
                                                 --learning_rate=5e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50

rm -rf models/*

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.challenge.train.csv' \
                                                --model_name="google/mt5-base" \
                                                --output_dir='models/ellipsis-type-mt5-challenge' \
                                                --hub_model_id='ellipsis-type-mt5-challenge' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=64 \
                                                --per_device_eval_batch_size=8 \
                                                --gradient_accumulation_steps=8 \
                                                --learning_rate=5e-5 \
                                                --eval_steps=50 \
                                                --model_type='encoder' \
                                                --min_count_per_class=70 \
                                                --max_count_per_class=5000 \
                                                --over_sample \
                                                --num_epochs=50

rm -rf models/*


python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.challenge.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --output_dir='ellipsis-discriminative-mt5-challenge' \
                                                 --hub_model_id='ellipsis-discriminative-mt5-challenge' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='discriminative' \
                                                 --learning_rate=5e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50

rm -rf models/*

python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.challenge.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --output_dir='models/ellipsis-extractive-mt5-challenge' \
                                                 --hub_model_id='ellipsis-extractive-mt5-challenge' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='extractive' \
                                                 --learning_rate=5e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50