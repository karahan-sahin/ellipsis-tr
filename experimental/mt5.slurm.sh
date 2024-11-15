#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --job-name=ellipsis-mt5
#SBATCH -o ../ellipsis-mt5.out
#SBATCH -t 2-0:00:00

source /opt/python3/venv/base/bin/activate

cd /users/karahan.sahin/ellipsis-tr
ls -ah
pip install -r requirements.txt

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.train.csv' \
                                                --model_name="google/mt5-base" \
                                                --hub_model_id='ellipsis-type-mt5' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=64 \
                                                --per_device_eval_batch_size=8 \
                                                --gradient_accumulation_steps=8 \
                                                --learning_rate=1e-5 \
                                                --eval_steps=100 \
                                                --model_type='encoder' \
                                                --min_count_per_class=70 \
                                                --max_count_per_class=5000 \
                                                --over_sample \
                                                --num_epochs=50

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --hub_model_id='ellipsis-discriminative-mt5' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='discriminative' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=100


python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --hub_model_id='ellipsis-extractive-mt5' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='extractive' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=100