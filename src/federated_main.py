#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from transformers import Wav2Vec2Processor
import pickle
import whisper
import pandas as pd
import json
from datasets import Dataset
from sklearn.cluster import KMeans
from transformers import Data2VecAudioConfig
from collections import Counter
import torch
import shutil

# import from home-made library
from options import args_parser
from utils import get_raw_dataset, exp_details, add_cluster_id, load_model, evaluateASR
from update_CPFL import update_network_weight, map_to_result
from training import client_train, centralized_training, client_getEmb

def FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, init_global_weights=None):
    global_weights_lst = []                                                                 # global_weights for all clusters
    for cluster_id in tqdm(range(args.num_lms)):                                            # train 1 cluster at a time
        model_id = cluster_id
        if args.num_lms != 1:                                                               # train with different cluster
            print(f'\n | Global Training Round for Cluster {cluster_id}: {epoch+1} |\n')    # print current round & cluster
        else:                                                                               # "only 1 cluster" means "no cluster"
            print(f'\n | Global Training Round: {epoch+1} |\n')                             # print current round
            cluster_id = None

        m = max(int(args.frac * args.num_users), 1)                                         # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)              # select by client_id
            
        local_weights_en = []                                                               # weight list for ASR encoder
        local_weights_de = []                                                               # weight list for ASR decoder
        num_training_samples_lst = []
        
        for idx in idxs_users:                                                              # for each client
            if init_global_weights != None:
                global_weights = init_global_weights[model_id]                              # get model weights of this cluster
            else:
                global_weights = None

            final_result = client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, idx,
                                                            epoch, cluster_id, global_weights)
                                                                                            # train from model in model_in_path + "_global/final/"
                                                                                            # or model in last round
                                                                                            # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_cluster" + str(cluster_id)
                                                                                            #   + ("_Training" + "Address") 

    return global_weights_lst                                                               # weight per cluster
    
# train 1 round for all clusters at once
def CBFL_training_clusters(args, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, Nth_Kmeans_update=None, init_weights_lst=None):
    if Nth_Kmeans_update == None:                                                           # no need to re-cluster
        epochs = args.epochs                                                                # perform training at once
    else:
        epochs = args.N_Kmeans_update                                                       # need to stop and update k-means model
    
    global_weights_lst = init_weights_lst
    for i in range(epochs):
        if Nth_Kmeans_update == None:                                                       # no need to re-cluster
            epoch = i                                                                       
        else:
            epoch = int(i + Nth_Kmeans_update*args.N_Kmeans_update)                         # current epoch     
        
        global_weights_lst = FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, global_weights_lst)
                                                                                            # model weights per cluster
        for j in range(args.num_lms):                                                       # for all cluster
            # aggregate model and save results
            global_weights = global_weights_lst[j]
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(epoch)
            model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update ASR weight
            model.save_pretrained(folder_to_save + "/final")

        model_in_path = args.model_in_path                                                  # keep model_in_path so that we can re-assign it later
        evaluateASR(args, epoch, test_dataset, train_dataset_supervised)                    # evaluate current models
        print(args.model_out_path+"#_CBFLASR_global_round" + str(epoch) + " evaluated.")
        for j in range(args.num_lms):
            shutil.rmtree(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(epoch))
                                                                                            # remove aggregated models
        # get ready for the next round
        args.model_in_path = model_in_path
        torch.cuda.empty_cache()
    return global_weights_lst


def get_clients_representations(args, model_in_path, train_dataset_supervised, test_dataset, TEST, cluster_id=None):
    multiprocessing.set_start_method('spawn', force=True)

    idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)     # select all clients
    pool = multiprocessing.Pool(processes=args.num_users)

    try:
        final_result = pool.starmap_async(client_getEmb, [(args, model_in_path, train_dataset_supervised, test_dataset, idx, cluster_id,
                                                            TEST) for idx in idxs_users])                                                                                                                                        
    except Exception as e:
        print(f"An error occurred while running local_model.update_weights(): {str(e)}")
    
    finally:
        final_result.wait()                                                                 # wait for all clients end
        results = final_result.get()                                                        # get results

    hidden_states_mean_lst = []                                                             # record hidden_states_mean from samples of all clients
    loss_lst = []                                                                           # record loss from samples of all clients
    entropy_lst = []
    vocab_ratio_rank_lst = []
    encoder_attention_1D_lst = []
    for idx in range(len(results)):                                                         # for each clients
        hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = results[idx]  
        hidden_states_mean_lst.extend(hidden_states_mean)                                   # save hidden_states_mean for this client
        loss_lst.extend(loss)
        entropy_lst.extend(entropy)
        vocab_ratio_rank_lst.extend(vocab_ratio_rank)
        encoder_attention_1D_lst.extend(encoder_attention_1D)
        # check dim
        #print("hidden_states_mean_lst: ", np.shape(hidden_states_mean_lst)) # [total num_samples, hidden_size]
        #print("loss_lst: ", np.shape(loss_lst)) # [total num_samples, 1]
        #print("entropy_lst: ", np.shape(entropy_lst)) # [total num_samples, 1]
    return hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst

CPFL_dataRoot = os.environ.get('CPFL_dataRoot')
def assign_cluster(args, dataset, kmeans, dataset_path, csv_path):
    torch.set_num_threads(1)
    
    # load ASR model
    mask_time_prob = 0                                                                      # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
    model = load_model(args, args.model_in_path, config)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

    # get emb.s ... 1 sample by 1 sample for dataset
    _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = map_to_result(dataset[0], processor, model, 0)
                                                                                            # start from 1st sample
    for i in range(len(dataset) - 1):
        _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2 = map_to_result(dataset[i+1], processor, model, i+1)
        
        hidden_states_mean.extend(hidden_states_mean_2)                                     # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
        loss.extend(loss2)                                                                  # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
        entropy.extend(entropy2)
        vocab_ratio_rank.extend(vocab_ratio_rank2)
        encoder_attention_1D.extend(encoder_attention_1D2)
        print("\r"+ str(i), end="")                                                         # show current file id
    
    # get cluster based on desired clustering metric
    #cluster_id_lst = kmeans.predict(entropy).tolist()                                      # result in list of cluster_id
    #cluster_id_lst = kmeans.predict(hidden_states_mean).tolist()
    cluster_id_lst = kmeans.predict(vocab_ratio_rank).tolist()
    #cluster_id_lst = kmeans.predict(encoder_attention_1D).tolist()

    counter = Counter(cluster_id_lst)                                                       # count number of samples of each cluster
    result = [counter[i] for i in range(max(cluster_id_lst) + 1)]
    print("cluster sample counts: ", result)                                                # show result

    # add to dataset
    dataset = dataset.map(lambda example: add_cluster_id(example, cluster_id_lst.pop(0)))
    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    
    dataset.save_to_disk(stored+"_temp")                                                    # save for later use
    if os.path.exists(stored):                                                              # path exist
        shutil.rmtree(stored)                                                               # remove previous data
    os.rename(stored+"_temp", stored)                                                       # rename as previous data
    
    print(csv_path.split("/")[-1].split(".")[0], " col_names: ", dataset.column_names)
    print("Dataset w/ cluster_id saved.")
    return dataset

def build_Kmeans_model(args, dx):
    kmeans = KMeans(n_clusters=args.num_lms)
    kmeans.fit(dx)
    pickle.dump(kmeans, open(args.Kmeans_model_path, 'wb'))                                 # save model for later use

    cluster_id_lst = kmeans.predict(dx).tolist()                                            # list of cluster_id
    counter = Counter(cluster_id_lst)                                                       # count overall number of samples of each cluster
    result = [counter[i] for i in range(max(cluster_id_lst) + 1)]
    print("overall cluster sample counts: ", result)                                        # show result

    path = './logs/Kmeans_log.txt'                                                          # save result
    f = open(path, 'a')
    f.write("---------------------------------------------------\n")
    f.write("Cluster centers: " + str(kmeans.cluster_centers_) + "\n")
    f.write("Overall cluster sample counts: " + str(result) + "\n")
    f.write("---------------------------------------------------\n")
    f.close()

    return kmeans

# FL stage 1: Global train ASR encoder & decoder
def GlobalTrainASR(args, train_dataset_supervised, test_dataset):                           # train from pretrain, final result in args.model_out_path + "_finetune" + "_global/final"
    args.local_ep = args.global_ep                                                          # use number of global epoch for global model
    args.STAGE = 0                                                                          # train ASR encoder & decoder
    centralized_training(args=args, model_in_path=args.pretrain_name, model_out_path=args.model_out_path+"_finetune", 
                            train_dataset=train_dataset_supervised, test_dataset=test_dataset, epoch=0)                     

# FL stage 3: perform k-means clustering
def Kmeans_clustering(args, train_dataset_supervised, test_dataset):
    # get clustering metric from all the client samples
    hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst = get_clients_representations(args=args, model_in_path=args.model_in_path, train_dataset_supervised=train_dataset_supervised,
                                                        test_dataset=test_dataset, TEST=False, cluster_id=None)
    # check dimension
    print("overall entropy_lst shape: ", np.shape(np.array(entropy_lst))) # (batch_size, 1)
    print("overall hidden_states_mean_lst shape: ", np.shape(np.array(hidden_states_mean_lst)))         # (batch_size, 1024)
    print("overall vocab_ratio_rank_lst shape: ", np.shape(np.array(vocab_ratio_rank_lst)))             # (batch_size, 32)
    print("overall encoder_attention_1D_lst shape: ", np.shape(np.array(encoder_attention_1D_lst)))     # (batch_size, ?)

    
    # Server selects best K centroid from above candidates
    #kmeans = build_Kmeans_model(args, entropy_lst)
    #kmeans = build_Kmeans_model(args, hidden_states_mean_lst)
    kmeans = build_Kmeans_model(args, vocab_ratio_rank_lst)
    #kmeans = build_Kmeans_model(args, encoder_attention_1D_lst)
    
    # add cluster_id to dataset
    args.dataset = 'adress'
    dataset_path = args.dataset_path_root + "/ADReSS_clustered/"                                        # new folder to save dataset
    assign_cluster(args, test_dataset, kmeans, dataset_path, csv_path= f"{CPFL_dataRoot}/mid_csv/test.csv")
    assign_cluster(args, train_dataset_supervised, kmeans, dataset_path, csv_path= f"{CPFL_dataRoot}/mid_csv/train.csv")
        
    
# FL stage 4: FL train ASR
def CPFL_TrainASR(args, train_dataset_supervised, test_dataset):
    global_weights_lst = None
    
    for i in range(int(args.epochs / args.N_Kmeans_update)):                                # "num of global rounds" / "num of rounds k-means model needs to be updated", e.g. int(10 / 5)
        if int(args.epochs / args.N_Kmeans_update) == 1:                                    # no cluster update
            Nth_Kmeans_update = None
        else:
            Nth_Kmeans_update = i
        global_weights_lst = CBFL_training_clusters(args=args, model_in_path_root=args.model_in_path+"_FLASR", model_out_path=args.model_out_path,
                                    train_dataset_supervised=train_dataset_supervised, test_dataset=test_dataset, Nth_Kmeans_update=Nth_Kmeans_update, 
                                                    init_weights_lst=global_weights_lst)
        # update global model for each cluster
        for j in range(args.num_lms):
            global_weights = global_weights_lst[j]                                          # get global weight for cluster j
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update weights
            model.save_pretrained(folder_to_save + "/final")


def get_dataset(args):                                                                      # return train_dataset_supervised, test_dataset
    args.dataset = "adress"                                                                 # get supervised dataset (adress)
    train_dataset_supervised, test_dataset = get_raw_dataset(args)                          # get dataset

    return train_dataset_supervised, test_dataset

if __name__ == '__main__':
    start_time = time.time()                                                                # record starting time

    path_project = os.path.abspath('..')                                                    # define paths
    
    args = args_parser()                                                                    # get configuration
    exp_details(args)                                                                       # print out details based on configuration
    
    train_dataset_supervised, test_dataset = get_dataset(args)

    # Training
    if args.FL_STAGE == 1:                                                              # train Fine-tuned ASR W_0^G
        print("| Start FL Training Stage 1 -- Global Train ASR |")
        args.STAGE = 0                                                                  # 0: train ASR encoder & decoder
        GlobalTrainASR(args=args, train_dataset_supervised=train_dataset_supervised, test_dataset=test_dataset)                      
                                                                                        # Train ASR encoder & decoder
        print("| FL Training Stage 1 Done|")
    elif args.FL_STAGE == 2:
        print("| Stage 2 Discarded!!|")
    elif args.FL_STAGE == 3:                                                            # K-means clustering
        print("| Start FL Training Stage 3|")
        Kmeans_clustering(args, train_dataset_supervised, test_dataset)
        print("| FL Training Stage 3 Done|")
    elif args.FL_STAGE == 4:                                                            # FL train ASR decoder
        print("| Start FL Training Stage 4|")
        args.STAGE = 0                                                                  # train ASR encoder as well
        CPFL_TrainASR(args, train_dataset_supervised, test_dataset)
        print("| FL Training Stage 4 Done|")
    else:
        print("Only FL Training Stage 1-4 is available, current FL_STAGE = ", args.FL_STAGE)
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
