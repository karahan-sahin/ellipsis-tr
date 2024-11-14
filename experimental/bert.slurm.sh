#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --job-name=ellipsis-bert
#SBATCH -o ../ellipsis-bert.out
#SBATCH -t 2-0:00:00

source /opt/python3/venv/base/bin/activate

cd /users/karahan.sahin/ellipsis-tr
ls -ah
pip install -r requirements.txt

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.train.csv' \
                                                --model_name="dbmdz/bert-base-turkish-cased" \
                                                --hub_model_id='ellipsis-type-bert' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=64 \
                                                --per_device_eval_batch_size=8 \
                                                --gradient_accumulation_steps=8 \
                                                --learning_rate=1e-5 \
                                                --eval_steps=200 \
                                                --train_size='mid_count' \
                                                --model_type='encoder' \
                                                --num_epochs=20

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="dbmdz/bert-base-turkish-cased" \
                                                 --hub_model_id='ellipsis-extractive-bert' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='extractive' \
                                                 --train_size='mid_count' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --eval_steps=200

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="dbmdz/bert-base-turkish-cased" \
                                                 --hub_model_id='ellipsis-discriminative-bert' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --gradient_accumulation_steps=8 \
                                                 --extraction_type='discriminative' \
                                                 --train_size='mid_count' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --eval_steps=200

