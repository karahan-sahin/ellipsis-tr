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
                                                --model_type='encoder'

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --hub_model_id='ellipsis-extractive-mt5' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --extraction_type='extractive' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="google/mt5-base" \
                                                 --hub_model_id='ellipsis-discriminative-mt5' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --extraction_type='discriminative' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500
