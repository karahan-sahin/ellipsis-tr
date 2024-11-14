#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --job-name=ellipsis-turna
#SBATCH -o ../ellipsis-turna.out
#SBATCH -t 2-0:00:00

source /opt/python3/venv/base/bin/activate

cd /users/karahan.sahin/ellipsis-tr
ls -ah
pip install -r requirements.txt

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.train.csv' \
                                                --model_name="boun-tabi-LMG/TURNA" \
                                                --hub_model_id='ellipsis-type-turna' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=4 \
                                                --per_device_eval_batch_size=2 \
                                                --gradient_accumulation_steps=32 \
                                                --eval_steps=500 \
                                                --model_type='encoder'

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="boun-tabi-LMG/TURNA" \
                                                 --hub_model_id='ellipsis-extractive-turna' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=4 \
                                                 --per_device_eval_batch_size=2 \
                                                 --gradient_accumulation_steps=32 \
                                                 --extraction_type='extractive' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500

python3 lib/training/run_span_classification.py  --dataset_file='data/ellipsis.span.train.csv' \
                                                 --model_name="boun-tabi-LMG/TURNA" \
                                                 --hub_model_id='ellipsis-discriminative-turna' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=4 \
                                                 --per_device_eval_batch_size=2 \
                                                 --gradient_accumulation_steps=32 \
                                                 --extraction_type='discriminative' \
                                                 --num_epochs=10 \
                                                 --eval_steps=500
