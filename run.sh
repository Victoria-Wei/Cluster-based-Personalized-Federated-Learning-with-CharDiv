#!/bin/bash

. ./path.sh
CUDA_VISIBLE_DEVICES=1 python src/federated_main.py --model=data2vec --gpu=1 --pretrain_name "./save/data2vec-audio-large-960h-local/" \
    --frac=1.0 --num_users=5 --global_ep=30 --learning_rate=1e-5 \
    --num_lms 7 --training_type 1 --local_ep 10 --epochs 10 --N_Kmeans_update 10 \
    --FL_STAGE 4 -model_out "./save/data2vec-audio-large-960h" -model_in "./save/data2vec-audio-large-960h" \
    -dataset_path_root "./datasets/" --eval_mode 2 --FL_type 1 --mu 0.01 --alpha 0.5 --beta 0.5 \
    #--train_batch_size 1 --eval_batch_size 1
    #--WeightedAvg --CBFL
    #2 --WeightedAvg
    #--train_batch_size 1 --eval_batch_size 1


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
