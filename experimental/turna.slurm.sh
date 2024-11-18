#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --job-name=ellipsis-turna-release
#SBATCH -o ../ellipsis-turna-release.out
#SBATCH -t 2-0:00:00

source /opt/python3/venv/base/bin/activate

cd /users/karahan.sahin/ellipsis-tr
ls -ah
pip install -r requirements.txt

python3 -m lib.training.run_type_classification --dataset_file='data/ellipsis.classification.release.train.csv' \
                                                --model_name="boun-tabi-LMG/TURNA" \
                                                --hub_model_id='ellipsis-type-turna-release' \
                                                --push_to_hub \
                                                --per_device_train_batch_size=4 \
                                                --per_device_eval_batch_size=2 \
                                                --gradient_accumulation_steps=64 \
                                                --learning_rate=1e-5 \
                                                --num_epochs=20 \
                                                --min_count_per_class=70 \
                                                --max_count_per_class=5000 \
                                                --over_sample \
                                                --eval_steps=50 \
                                                --model_type='encoder'

python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.release.train.csv' \
                                                 --model_name="boun-tabi-LMG/TURNA" \
                                                 --hub_model_id='ellipsis-extractive-turna-release' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=4 \
                                                 --per_device_eval_batch_size=2 \
                                                 --gradient_accumulation_steps=64 \
                                                 --extraction_type='extractive' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50

python3 -m lib.training.run_span_classification  --dataset_file='data/ellipsis.span.release.train.csv' \
                                                 --model_name="boun-tabi-LMG/TURNA" \
                                                 --hub_model_id='ellipsis-discriminative-turna-release' \
                                                 --push_to_hub \
                                                 --per_device_train_batch_size=4 \
                                                 --per_device_eval_batch_size=2 \
                                                 --gradient_accumulation_steps=64 \
                                                 --extraction_type='discriminative' \
                                                 --learning_rate=1e-5 \
                                                 --num_epochs=20 \
                                                 --min_count_per_class=70 \
                                                 --max_count_per_class=5000 \
                                                 --over_sample \
                                                 --eval_steps=50
