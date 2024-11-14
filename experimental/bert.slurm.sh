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
                                                --eval_steps=500 \
                                                --train_size='mid_count' \
                                                --model_type='encoder'

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="dbmdz/bert-base-turkish-cased" \
                                                 --hub_model_id='ellipsis-extractive-bert' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --extraction_type='extractive' \
                                                 --train_size='mid_count' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="dbmdz/bert-base-turkish-cased" \
                                                 --hub_model_id='ellipsis-discriminative-bert' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=64 \
                                                 --per_device_eval_batch_size=8 \
                                                 --extraction_type='discriminative' \
                                                 --train_size='mid_count' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500
