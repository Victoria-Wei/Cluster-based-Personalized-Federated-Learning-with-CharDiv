#!/bin/bash

. ./path.sh
# FL_STAGE 1, 2, 4
#CUDA_VISIBLE_DEVICES=0 python src/federated_main.py --model=data2vec --gpu=1 --pretrain_name "./save/data2vec-audio-large-960h-local/" \
#    --frac=1.0 --num_users=5 --global_ep=30 --learning_rate=1e-5 \
#    --num_lms 1 --training_type 3 --local_ep 10 --epochs 10 --N_Kmeans_update 10 \
#    --FL_STAGE 2 -model_out "./save/data2vec-audio-large-960h" -model_in "./save/data2vec-audio-large-960h"
#    --train_batch_size 1 --eval_batch_size 1 \
# 第1行：不改的
# 第2行：可以改，但基本上不改
# 第3行：可能會改的
# 第4行：一定要改

# FL_STAGE 3
#CUDA_VISIBLE_DEVICES=0 python src/federated_main.py --model=data2vec --gpu=1 --pretrain_name "./save/data2vec-audio-large-960h-local/" \
#    --frac=1.0 --num_users=5 --global_ep=30 --learning_rate=1e-4 \
#    --num_lms 7 --training_type 1 --local_ep 10 --epochs 10 --N_Kmeans_update 10 \
#    --FL_STAGE 3 -model_out "/mnt/External/Seagate/weitung/save/data2vec-audio-large-960h" -model_in "/mnt/External/Seagate/weitung/save/data2vec-audio-large-960h_FLASR_global" \
#    -dataset_path_root "./dataset" --eval_mode 3
    
# FL_STAGE 4: 跟前面一樣，只是現在放在stage 3之後
CUDA_VISIBLE_DEVICES=1 python src/federated_main.py --model=data2vec --gpu=1 --pretrain_name "./save/data2vec-audio-large-960h-local/" \
    --frac=1.0 --num_users=5 --global_ep=30 --learning_rate=1e-5 \
    --num_lms 1 --training_type 1 --local_ep 10 --epochs 10 --N_Kmeans_update 10 \
    --FL_STAGE 4 -model_out "/mnt/External/Seagate/weitung/save/data2vec-audio-large-960h" -model_in "/mnt/External/Seagate/weitung/save/data2vec-audio-large-960h" \
    -dataset_path_root "/mnt/External/Seagate/weitung/datasets/Exp10_datasets/" --eval_mode 2 --FL_type 3 --mu 0.01 --alpha 0.5 --beta 0.5 \
    --train_batch_size 1 --eval_batch_size 1
    #--WeightedAvg --CBFL
    #2 --WeightedAvg
    #--train_batch_size 1 --eval_batch_size 1
# Extract Embs
#bash run_extract.sh
#bash run_extract.sh
# model=
#CUDA_VISIBLE_DEVICES=0 python src/federated_main.py --model=data2vec --gpu=1 --pretrain_name "facebook/data2vec-audio-large-960h"\
#    --frac=1.0 --num_users=2 --global_ep=30 --learning_rate=1e-5 \
#    --training_type 5 --local_ep 10 --epochs 10 --EXTRACT --FL_STAGE 4 \
#    --num_lms 5 -model_in "./save/data2vec-audio-large-960h" -csv "data2vec-audio-large-960h_CBFLASR"
    # 一次取多個cluster： --num_lms 5 -model_in "./save/data2vec-audio-large-960h#_CBFLASR_global" -csv "data2vec-audio-large-960h_CBFLASR"
    # 單純取特定model： --num_lms 1 -model_in "./save/data2vec-audio-large-960h_(模型名稱)" -csv "data2vec-audio-large-960h_(模型名稱)"
# generate detail WER
#mkdir "./results/detail_wer/data2vec-audio-large-960h_CBFLASR/"
#python src/detail_wer.py -v 3 -csv "./results/data2vec-audio-large-960h_CBFLASR.csv" -save "./results/detail_wer/data2vec-audio-large-960h_CBFLASR/" -T

#CUDA_VISIBLE_DEVICES=0 python src/federated_main.py --model=data2vec --dataset=adress --gpu=1 --pretrain_name "facebook/data2vec-audio-large-960h"\
#   -model_out "./save/data2vec-audio-large-960h_new1_recall" -log "data2vec-audio-large-960h_new1_recall_FL.txt"\
#   --AD_loss "recall" --frac=1.0 \
#   --local_ep 10 --epochs=10 --num_users=2 \
#   --FL_STAGE 1 --eval_steps 1000 --supervised_level 0.5 \
#   --data_combine_type "separated"
    # 用FL_STAGE取代STAGE
    
    #-csv "data2vec-audio-large-960h_new1_recall" (need to be checked)
    #-model_in "/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_new1_recall/final/" \

#python src/federated_main.py --model=data2vec --dataset=adress --gpu=1 --pretrain_name "facebook/data2vec-audio-large-960h"\
#    -model_in "./save/data2vec-audio-large-960h_new1_recall" \
#    -model_out "./save/data2vec-audio-large-960h_new2_recall" -log "data2vec-audio-large-960h_new2_recall_FL.txt" \
#    --AD_loss "recall" --frac=1.0  \
#    --local_ep 5 --epochs=2 --num_users=2 \
#    --FL_STAGE 2 # 用FL_STAGE取代STAGE
    # -csv "data2vec-audio-large-960h_new2_recall_FL" (need to be checked)

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
