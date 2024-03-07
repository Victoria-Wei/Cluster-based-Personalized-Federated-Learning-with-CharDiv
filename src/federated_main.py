#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from tqdm import tqdm

from options import args_parser
from utils import get_raw_dataset, average_weights, exp_details

import multiprocessing
from update_CBFL import update_network_weight, get_model_weight

from training import client_train, centralized_training, client_getEmb
from utils import prepare_dataset, TextNor
from transformers import Wav2Vec2Processor

import pickle
import whisper
import pandas as pd
import json
from datasets import Dataset
from sklearn.cluster import KMeans

from transformers import Data2VecAudioConfig
from models import Data2VecAudioForCTC_CBFL
from update_CBFL import map_to_result
from utils import add_cluster_id, reorder_col, add_entropy, get_overall_wer, load_model, save_weights, evaluateASR
from collections import Counter
import torch
import shutil
from utils import train_split_supervised, concatenate_ds

class TeacherStudentLearning:
    def __init__(self, loadpath=None, savepath=None, load_mdl='large-v2'):
        self.transcript = []
        self.loadpath = loadpath
        self.savepath = savepath
        self.DACS_dataRoot = '/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train'
        out_root='/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis'
        out_dirname='transcript_whisper'
        out_filename='train.csv'
        self.out_file=f"{out_root}/{out_dirname}/{out_filename}"
        self.model = whisper.load_model(load_mdl)            
    def add_transcript_to_dataset(self, dataset, transcript_in):
        dataset = dataset.add_column("text", transcript_in)
        return dataset
    
    def save_transcript(self, dataset,outFile):
        df = pd.DataFrame(dataset)
        df.to_csv(outFile, index=False)
    def load_transcript(self, in_file):
        df = pd.read_csv(in_file)
        dataset = Dataset.from_pandas(df)
        return dataset
    def transcribe(self, dataset):
        transcript=[]
        for i,batch in tqdm(enumerate(dataset)):
            singleFile=batch['path']
            file=f'{self.DACS_dataRoot}/clips/{singleFile}'
            result = self.model.transcribe(file, language="en")
            pred_text=result['text'].upper().strip()
            print(pred_text)
            transcript.append(pred_text)
        return transcript
    def transcribe_n_Merge(self, dataset):
        transcript = self.transcribe(dataset)
        ds=self.add_transcript_to_dataset(self, dataset, transcript)
        return ds
    def FilterAvailAudios(self, dataset):
            dataset_enoughLen = dataset.filter(lambda example: len(example['array']) >= 1600)
            dataset_enoughLen_enoughtext = dataset_enoughLen.filter(lambda example: len(example['text']) > 0)
            return dataset_enoughLen_enoughtext

def FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, init_global_weights=None):
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

        if args.chosen_clients == True:
            idxs_users = [0, 4]
            print("||    Perform training on clients: ", idxs_users, "    ||")
            
        local_weights_en = []                                                               # weight list for ASR encoder
        local_weights_de = []                                                               # weight list for ASR decoder
        num_training_samples_lst = []
        
        for idx in idxs_users:
            #idx = 4
            # all client of certain cluster perform training
            if init_global_weights != None:
                global_weights = init_global_weights[model_id]
            else:
                global_weights = None


            final_result = client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx,
                                                            epoch, cluster_id, global_weights)
                                                                                            # train from model in model_in_path + "_global/final/"
                                                                                            # or model in last round
                                                                                            # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_cluster" + str(cluster_id)
                                                                                            #   + ("_Training" + "Address") or ("_Training" + "AddressoWhisper") or ("_Training" + "AddressoWhisperandAddress")                                                                                                                                 
    # """    
            #aaa=ccc
            w, num_training_samples = final_result                                          # function client_train returns w, num_training_samples
            local_weights_en.append(copy.deepcopy(w[0]))                                    # save encoder weight for this client
            local_weights_de.append(copy.deepcopy(w[1]))                                    # save decoder weight for this client
            num_training_samples_lst.append(num_training_samples)                           # save num of training samples

        # aggregate weights of encoder and decoder
        global_weights = [average_weights(local_weights_en, num_training_samples_lst, args.WeightedAvg), average_weights(local_weights_de, num_training_samples_lst, args.WeightedAvg)]
        global_weights_lst.append(global_weights)
    # """
    # print("End of fine-tuning...")
    # aaa=ccc
    return global_weights_lst # 每個cluster一個weight
    

# use for-loop to save memory usage(?)
def FL_training_rounds_loop(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update=None, global_weights=None):
    if Nth_Kmeans_update == None:                                                           # no need to re-cluster
        epochs = args.epochs                                                                # perform training at once
    else:
        epochs = args.N_Kmeans_update                                                       # need to stop and update k-means model

    for i in tqdm(range(epochs)):                                                           # train for given global rounds
        if Nth_Kmeans_update != None:                                                       # re-clustering is needed
            epoch = int(i + Nth_Kmeans_update*args.N_Kmeans_update)                         # assign starting round
        else:
            epoch = i

        if cluster_id != None:  
            print(f'\n | Global Training Round for Cluster {cluster_id}: {epoch+1} |\n')    # print current round & cluster
        else:
            print(f'\n | Global Training Round: {epoch+1} |\n')                             # print current round

        m = max(int(args.frac * args.num_users), 1)                                         # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)              # select by client_id

        local_weights_en = []                                                               # weight list for ASR encoder
        local_weights_de = []                                                               # weight list for ASR decoder
        num_training_samples_lst = []
        for idx in idxs_users:
            # all client of certain cluster perform training
            final_result = client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx,
                                                            epoch, cluster_id, global_weights)
                                                                                            # train from model in model_in_path + "_global/final/"
                                                                                            # or model in last round
                                                                                            # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_cluster" + str(cluster_id)
                                                                                            #   + ("_Training" + "Address") or ("_Training" + "AddressoWhisper") or ("_Training" + "AddressoWhisperandAddress")                                                                                                                                 
        
            w, num_training_samples = final_result                                          # function client_train returns w, num_training_samples
            local_weights_en.append(copy.deepcopy(w[0]))                                    # save encoder weight for this client
            local_weights_de.append(copy.deepcopy(w[1]))                                    # save decoder weight for this client
            num_training_samples_lst.append(num_training_samples)                           # save num of training samples

        # aggregate weights of encoder and decoder
        global_weights = [average_weights(local_weights_en, num_training_samples_lst, args.WeightedAvg), average_weights(local_weights_de, num_training_samples_lst, args.WeightedAvg)]

    return global_weights

def FL_training_rounds(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update=None, global_weights=None):
    multiprocessing.set_start_method('spawn', force=True)
    if Nth_Kmeans_update == None:                                                           # no need to re-cluster
        epochs = args.epochs                                                                # perform training at once
    else:
        epochs = args.N_Kmeans_update                                                       # need to stop and update k-means model

    for i in tqdm(range(epochs)):                                                           # train for given global rounds
        if Nth_Kmeans_update != None:                                                       # re-clustering is needed
            epoch = int(i + Nth_Kmeans_update*args.N_Kmeans_update)                         # assign starting round
        else:
            epoch = i

        if cluster_id != None:  
            print(f'\n | Global Training Round for Cluster {cluster_id}: {epoch+1} |\n')    # print current round & cluster
        else:
            print(f'\n | Global Training Round: {epoch+1} |\n')                             # print current round

        m = max(int(args.frac * args.num_users), 1)                                         # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)              # select by client_id
        pool = multiprocessing.Pool(processes=m)

        try:
            # all client of certain cluster perform training
            final_result = pool.starmap_async(client_train, [(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx,
                                                            epoch, cluster_id, global_weights) for idx in idxs_users])
                                                                                            # train from model in model_in_path + "_global/final/"
                                                                                            # or model in last round
                                                                                            # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_cluster" + str(cluster_id)
                                                                                            #   + ("_Training" + "Address") or ("_Training" + "AddressoWhisper") or ("_Training" + "AddressoWhisperandAddress")                                                                                                                                 
        except Exception as e:
            print(f"An error occurred while running local_model.update_weights(): {str(e)}")
        
        finally:
            final_result.wait()                                                             # wait for all clients end
            results = final_result.get()                                                    # get results
        
        local_weights_en = []                                                               # weight list for ASR encoder
        local_weights_de = []                                                               # weight list for ASR decoder
        num_training_samples_lst = []
        for idx in range(len(results)):                                                     # for each participated clients
            w, num_training_samples = results[idx]                                          # function client_train returns w, num_training_samples
            local_weights_en.append(copy.deepcopy(w[0]))                                    # save encoder weight for this client
            local_weights_de.append(copy.deepcopy(w[1]))                                    # save decoder weight for this client
            num_training_samples_lst.append(num_training_samples)                           # save num of training samples

        # aggregate weights of encoder and decoder
        global_weights = [average_weights(local_weights_en, num_training_samples_lst, args.WeightedAvg), average_weights(local_weights_de, num_training_samples_lst, args.WeightedAvg)]

    return global_weights

def CBFL_training_rounds(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_num, Nth_Kmeans_update=None, init_weights_lst=None):
    global_weights_lst = []
    if cluster_num == 1:                                                                    # only 1 cluster
        cluster_id = None                                                                   # train as a whole
        #global_weights = FL_training_rounds(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update)
        global_weights = FL_training_rounds_loop(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update)
        global_weights_lst.append(global_weights) 

    else:
        for k in range(cluster_num):                                                        # train cluster by cluster
            cluster_id = k                                                                  # k as cluster_id
            if init_weights_lst != None:                                                    # if given initial weight
                global_weights = init_weights_lst[k]                                        # assign as global weights
            else:
                global_weights = None
            #global_weights = FL_training_rounds(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update, global_weights)
            global_weights = FL_training_rounds_loop(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, cluster_id, Nth_Kmeans_update, global_weights)
            global_weights_lst.append(global_weights)                                       # record final aggregated weights for each cluster
    
    return global_weights_lst

# train 1 round for all clusters at once
def CBFL_training_clusters(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, Nth_Kmeans_update=None, init_weights_lst=None):
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
        
        global_weights_lst = FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, global_weights_lst)
        
                                                                                            # 每個cluster一個weight
        for j in range(args.num_lms):                                                       # for all cluster
            # aggregate model and save results
            global_weights = global_weights_lst[j]
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(epoch)
            if args.STAGE == 1: # freeze all, train ASR decoder alone
                save_weights(folder_to_save, global_weights)
            else:
                model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update decoder
                model.save_pretrained(folder_to_save + "/final")

        model_in_path = args.model_in_path                                                  # keep model_in_path so that we can re-assign it later
        evaluateASR(args, epoch, test_dataset, train_dataset_supervised)
        print(args.model_out_path+"#_CBFLASR_global_round" + str(epoch) + " evaluated.")
        for j in range(args.num_lms):
            shutil.rmtree(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(epoch))
                                                                                            # remove aggregated results
        # get ready for the next round
        args.model_in_path = model_in_path
        torch.cuda.empty_cache()
    return global_weights_lst

# use for-loop to save memory usage
def get_clients_representations_loop(args, model_in_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, TEST, cluster_id=None):
    idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)     # select all clients

    hidden_states_mean_lst = []                                                             # record hidden_states_mean from samples of all clients
    loss_lst = []                                                                           # record loss from samples of all clients
    entropy_lst = []
    vocab_ratio_rank_lst = []
    encoder_attention_1D_lst = []
    for idx in idxs_users:
        final_result = client_getEmb(args, model_in_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx, cluster_id,
                                                            TEST)                                                                                                                          

        hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = final_result                                    # [num_samples, hidden_size]
        hidden_states_mean_lst.extend(hidden_states_mean)                                   # save hidden_states_mean for this client
        loss_lst.extend(loss)
        entropy_lst.extend(entropy)
        vocab_ratio_rank_lst.extend(vocab_ratio_rank)
        encoder_attention_1D_lst.extend(encoder_attention_1D)
        #print("hidden_states_mean_lst: ", np.shape(hidden_states_mean_lst)) # [total num_samples, hidden_size]
        #print("loss_lst: ", np.shape(loss_lst)) # [total num_samples, 1]
        #print("entropy_lst: ", np.shape(entropy_lst)) # [total num_samples, 1]
    return hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst                   # [total num_samples, hidden_size], [total num_samples, 1], [total num_samples, 1]


def get_clients_representations(args, model_in_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, TEST, cluster_id=None):
    multiprocessing.set_start_method('spawn', force=True)

    idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)     # select all clients
    pool = multiprocessing.Pool(processes=args.num_users)

    try:
        final_result = pool.starmap_async(client_getEmb, [(args, model_in_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx, cluster_id,
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
        hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = results[idx]                                             # [num_samples, hidden_size]
        hidden_states_mean_lst.extend(hidden_states_mean)                                   # save hidden_states_mean for this client
        loss_lst.extend(loss)
        entropy_lst.extend(entropy)
        vocab_ratio_rank_lst.extend(vocab_ratio_rank)
        encoder_attention_1D_lst.extend(encoder_attention_1D)
        #print("hidden_states_mean_lst: ", np.shape(hidden_states_mean_lst)) # [total num_samples, hidden_size]
        #print("loss_lst: ", np.shape(loss_lst)) # [total num_samples, 1]
        #print("entropy_lst: ", np.shape(entropy_lst)) # [total num_samples, 1]
    return hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst                                                # [total num_samples, hidden_size], [total num_samples, 1], [total num_samples, 1]

def getTestsetEmb(args, model_in_path, test_dataset, save_csv):
    multiprocessing.set_start_method('spawn', force=True)

    pool = multiprocessing.Pool(processes=args.num_lms)                                     # for each cluster

    try:
        if args.num_lms > 1:
            txt = model_in_path.split("#")
            final_result = pool.starmap_async(client_getEmb, [(args, txt[0] + "_cluster" + str(idx) + txt[1], None, None, test_dataset, None, idx,
                                                            True) for idx in range(args.num_lms)])
                                                                                            # extract embs of cluster k using model k
        else:
            final_result = pool.starmap_async(client_getEmb, [(args, model_in_path, None, None, test_dataset, None, None, True)])
    except Exception as e:
        print(f"An error occurred while running local_model.update_weights(): {str(e)}")
    
    finally:
        final_result.wait()                                                                 # wait for all clients end
        results = final_result.get()                                                        # get results
    aaa=cc
    df = pd.DataFrame()                                                                     # record df from samples of all clusters
    for idx in range(len(results)):                                                         # for each cluster
        df2 = results[idx] 
        df = pd.concat([df, df2], ignore_index=True)

    if save_csv:
        csv_path = "./results/" + args.csv_path + ".csv"
        df.to_csv(csv_path)
        print("Testing data Saved.")

DACS_dataRoot = os.environ.get('DACS_dataRoot')
def assign_cluster(args, dataset, kmeans, dataset_path, csv_path, target_dataset=None):
    torch.set_num_threads(1)
    # load ASR model
    mask_time_prob = 0                                                                      # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
    if args.FL_STAGE == 3:
        #model = Data2VecAudioForCTC_CBFL.from_pretrained(args.model_in_path+"/final/", config=config, args=args)
        model = load_model(args, args.model_in_path, config)
    else:
        model_lst = []
        for cluster_id in range(args.num_lms):
            txt = args.model_in_path.split("#")
            #model = Data2VecAudioForCTC_CBFL.from_pretrained(txt[0] + "_cluster" + str(cluster_id) + txt[1]+"/final/", config=config, args=args)
            model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
            model_lst.append(model)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

    # get emb.s ... 1 sample by 1 sample for dataset
    if args.FL_STAGE == 3:
        _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = map_to_result(dataset[0], processor, model, 0)
    else:
        _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = map_to_result(dataset[0], processor, model_lst[dataset[0]["cluster_id"]], 0)
                                                                                            # choose corresponding model
    for i in range(len(dataset) - 1):
        if args.FL_STAGE == 3:
            _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2 = map_to_result(dataset[i+1], processor, model, i+1)
        else:
            _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2 = map_to_result(dataset[i+1], processor, model_lst[dataset[i+1]["cluster_id"]], i+1)
                                                                                            # choose corresponding model        
        hidden_states_mean.extend(hidden_states_mean_2)                                     # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
        loss.extend(loss2)                                                                  # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
        entropy.extend(entropy2)
        vocab_ratio_rank.extend(vocab_ratio_rank2)
        encoder_attention_1D.extend(encoder_attention_1D2)
        print("\r"+ str(i), end="")
    
    # get cluster
    #cluster_id_lst = kmeans.predict(entropy).tolist()                                       # list of cluster_id
    #cluster_id_lst = kmeans.predict(hidden_states_mean).tolist()
    cluster_id_lst = kmeans.predict(vocab_ratio_rank).tolist()
    #cluster_id_lst = kmeans.predict(encoder_attention_1D).tolist()

    counter = Counter(cluster_id_lst)                                                       # count number of samples of each cluster
    result = [counter[i] for i in range(max(cluster_id_lst) + 1)]
    print("cluster sample counts: ", result)                                                # show result

    # add to dataset
    dataset = dataset.map(lambda example: add_cluster_id(example, cluster_id_lst.pop(0)))
    #print("dataset entropy_lst shape: ", np.shape(np.array(entropy)))
    """
    if "entropy_history" not in dataset.column_names:                                       # no entropy_history exist
        dataset = dataset.map(lambda example: add_entropy(example, entropy.pop(0), first_time=True)) 
                                                                                            # create new list
    else:
        dataset = dataset.map(lambda example: add_entropy(example, entropy.pop(0), first_time=False))  # add to existing list
    """
    if target_dataset != None: 
        dataset = reorder_col(target_dataset, dataset)                                      # reorder as target_dataset

    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    
    #print("dataset (to-be saved) entropy_lst shape: ", np.shape(np.array(dataset[0]["entropy_history"]))) # (舊的+1, )
    dataset.save_to_disk(stored+"_temp")                                                            # save for later use
    if os.path.exists(stored):                                                              # path exist
        shutil.rmtree(stored)                                                               # remove previous data
    os.rename(stored+"_temp", stored)                                                       # rename as previous data
    
    print(csv_path.split("/")[-1].split(".")[0], " col_names: ", dataset.column_names)
    print("Dataset w/ cluster_id saved.")
    return dataset

def build_Kmeans_model(args, dx):
    #model = pickle.load(open(args.Kmeans_model_path, 'rb'))
    #if args.FL_STAGE == 4:                                                                  # need centers from last k-means model
    #    kmeans_last = pickle.load(open(args.Kmeans_model_path, 'rb'))                       # load the last k-means model
    #    cluster_centers1 = kmeans_last.cluster_centers_                                     # get centers
    #    cluster_centers1_new = np.concatenate((cluster_centers1, [[0]]*len(cluster_centers1)), axis=1) 
    #                                                                                        # 新維度皆設為0，當作一開始的center
    #    kmeans = KMeans(n_clusters=args.num_lms, init=cluster_centers1_new)
    #else:
    kmeans = KMeans(n_clusters=args.num_lms)
    kmeans.fit(dx)
    pickle.dump(kmeans, open(args.Kmeans_model_path, 'wb'))                                 # save model for later use

    cluster_id_lst = kmeans.predict(dx).tolist()                                            # list of cluster_id
    counter = Counter(cluster_id_lst)                                                       # count overall number of samples of each cluster
    result = [counter[i] for i in range(max(cluster_id_lst) + 1)]
    print("overall cluster sample counts: ", result)                                        # show result

    path = './logs/Kmeans_log.txt'
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

# FL stage 2: FL train ASR encoder & decoder for 1 round
# train from args.model_in_path, final result in args.model_out_path + "_finetune" + "_client" + str(self.client_id) + "_round" + str(global_round) + "_cluster" + str(self.cluster_id) + "_Training" + "Address"
#                                             or args.model_out_path + "_finetune" + "_client" + str(self.client_id) + "_round" + str(global_round) + "_cluster" + str(self.cluster_id) + "_Training" + "AddressoWhisper"
#                                             or args.model_out_path + "_finetune" + "_client" + str(self.client_id) + "_round" + str(global_round) + "_cluster" + str(self.cluster_id) + "_Training" + "AddressoWhisper&Address"
# aggregated model in args.model_out_path+"_FLASR_global/final"
def FL_TrainASR(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset):
    args.epochs = 1                                                                         # for 1 round only
    global_weights_lst = CBFL_training_rounds(args=args, model_in_path_root=args.model_in_path+"_finetune", model_out_path=args.model_out_path+"_finetune",
                                        train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
                                        test_dataset=test_dataset, cluster_num=1)
    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_finetune_global/final/", target_weight=global_weights_lst[0], network="ASR") 
    model.save_pretrained(args.model_out_path+"_FLASR_global/final")

# FL stage 3: perform k-means clustering
def Kmeans_clustering(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset):
    # 每個training utt（sample）的embs.平均over T time-steps，回傳所有sample的平均（即"有N個samples就會回傳N個averaged embs."的意思）
    if args.FL_STAGE == 3:
        hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst = get_clients_representations(args=args, model_in_path=args.model_in_path, train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised, 
                                                            test_dataset=test_dataset, TEST=False, cluster_id=None)
    else:
        entropy_lst = []
        hidden_states_mean_lst = []
        vocab_ratio_rank_lst = []
        encoder_attention_1D_lst = []
        for i in range(args.num_lms):                           # cluster by cluster
            # args.model_in_path = args.model_out_path+"#_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            txt = args.model_in_path.split("#")
            hidden_states_mean_cur, loss_lst, entropy_cur, vocab_ratio_rank_cur, encoder_attention_1D_cur = get_clients_representations(args=args, model_in_path=txt[0] + "_cluster" + str(i) + txt[1], train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised, 
                                                                test_dataset=test_dataset, TEST=False, cluster_id=i)
            entropy_lst.extend(entropy_cur)
            hidden_states_mean_lst.extend(hidden_states_mean_cur)
            vocab_ratio_rank_lst.extend(vocab_ratio_rank_cur)
            encoder_attention_1D_lst.extend(encoder_attention_1D_cur)
        # 加上前一次的entropy

    #print(loss_lst)
    print("overall entropy_lst shape: ", np.shape(np.array(entropy_lst))) # 應該包含最新的entropy及以往的（如果有的話）, (batch_size, 2)
    print("overall hidden_states_mean_lst shape: ", np.shape(np.array(hidden_states_mean_lst))) # 應該包含最新的entropy及以往的（如果有的話）, (batch_size, 2)
    print("overall vocab_ratio_rank_lst shape: ", np.shape(np.array(vocab_ratio_rank_lst)))
    print("overall encoder_attention_1D_lst shape: ", np.shape(np.array(encoder_attention_1D_lst)))

    
    # Server selects best K centroid from above candidates
    #kmeans = build_Kmeans_model(args, entropy_lst)
    #kmeans = build_Kmeans_model(args, hidden_states_mean_lst)
    kmeans = build_Kmeans_model(args, vocab_ratio_rank_lst)
    #kmeans = build_Kmeans_model(args, encoder_attention_1D_lst)
    
    # add cluster_id to dataset
    args.dataset = 'adress'
    dataset_path = args.dataset_path_root + "/ADReSS_clustered/"
    assign_cluster(args, test_dataset, kmeans, dataset_path, csv_path= f"{DACS_dataRoot}/mid_csv/test.csv")
    if train_dataset_supervised != None:
        train_dataset_supervised = assign_cluster(args, train_dataset_supervised, kmeans, dataset_path, csv_path= f"{DACS_dataRoot}/mid_csv/train.csv")
        
    if train_dataset_unsupervised != None:
        args.dataset = 'adresso'
        assign_cluster(args, train_dataset_unsupervised, kmeans, dataset_path=args.dataset_path_root + "/ADReSSo_clustered/",
                        csv_path= "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/train_ADReSSo.csv",
                        target_dataset=train_dataset_supervised)
    
# FL stage 4: FL train ASR Decoder
def FL_TrainASRdecoder(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset):
    global_weights_lst = None
    # samples of cluster k train model k
    for i in range(int(args.epochs / args.N_Kmeans_update)):                                # "num of global rounds" / "num of rounds k-means model needs to be updated", e.g. int(10 / 5)
        # 一次一段，做完更新cluster
        #global_weights_lst = CBFL_training_rounds(args=args, model_in_path_root=args.model_in_path+"_FLASR", model_out_path=args.model_out_path,
        #                                    train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
        #                                    test_dataset=test_dataset, cluster_num=args.num_lms, Nth_Kmeans_update=i, init_weights_lst=global_weights_lst)
                                                                                            # 回傳所有cluster的weight
        if int(args.epochs / args.N_Kmeans_update) == 1: # 不分段
            Nth_Kmeans_update = None
        else:
            Nth_Kmeans_update = i
        global_weights_lst = CBFL_training_clusters(args=args, model_in_path_root=args.model_in_path+"_FLASR", model_out_path=args.model_out_path,
                                    train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
                                    test_dataset=test_dataset, Nth_Kmeans_update=Nth_Kmeans_update, init_weights_lst=global_weights_lst)
        # update global model for each cluster
        for j in range(args.num_lms):
            global_weights = global_weights_lst[j]
            #model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update decoder 
            #model.save_pretrained(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1)) + "/final")
                                                                                            # save aggregated model for each cluster
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            if args.STAGE == 1: # freeze all, train ASR decoder alone
                save_weights(folder_to_save, global_weights)
            else:
                model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update decoder
                model.save_pretrained(folder_to_save + "/final")
        if i < (int(args.epochs / args.N_Kmeans_update) - 1):                               # update cluster except for the last round
            model_in_path = args.model_in_path
            args.model_in_path = args.model_out_path+"#_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            Kmeans_clustering(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
                                                                                            # re-cluster
            # re-load dataset
            train_dataset_supervised, train_dataset_unsupervised, test_dataset = get_dataset(args)
            args.model_in_path = model_in_path
            for j in range(args.num_lms):
                shutil.rmtree(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1)))
                                                                                                # remove aggregated results

def stage4_debug(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset):
    #############
    # debug
    #############
    ##############################################################################
    # data2vec-audio-large-960h_cluster0_CBFLASR_global_roundN的結果已經訓練完畢
    ##############################################################################
    """
    epoch = 9 # N
    model_in_path = args.model_in_path                                                  # keep model_in_path so that we can re-assign it later
    evaluateASR(args, epoch, test_dataset)
    print(args.model_out_path+"#_CBFLASR_global_round" + str(epoch) + " evaluated.")
                                                                                        # remove aggregated results
    # get ready for the next round
    args.model_in_path = model_in_path
    """
    ###########################################################################################################################################################
    ##############################################################################
    # data2vec-audio-large-960h_cluster0_CBFLASR_global_round1的結果已經evaluate完畢
    ##############################################################################
    
    i = 1
    model_in_path = args.model_in_path
    args.model_in_path = args.model_out_path+"#_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
    Kmeans_clustering(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
                                                                                    # re-cluster
    # re-load dataset
    train_dataset_supervised, train_dataset_unsupervised, test_dataset = get_dataset(args)
    args.model_in_path = model_in_path
                                                                                        # remove aggregated results
    ###########################################################################################################################################################
    # 把round N aggregate的結果load進來
    global_weights_lst = []
    rnd = 3 # N
    for i in range(args.num_lms): # for each cluster
        source_path = args.model_out_path+"_cluster" + str(i) + "_CBFLASR_global_round" + str(rnd)
        w, _ = get_model_weight(args, source_path, "ASR")                                       # function client_train returns w, num_training_samples
        global_weights_lst.append(copy.deepcopy(w))

    for j in range(args.num_lms):
        shutil.rmtree(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(rnd))

    # 繼續train round 4-9
    i_range = [2, 3, 4]
    for i in i_range:                                # "num of global rounds" / "num of rounds k-means model needs to be updated", e.g. int(10 / 5)

        if int(args.epochs / args.N_Kmeans_update) == 1: # 不分段
            Nth_Kmeans_update = None
        else:
            Nth_Kmeans_update = i
        global_weights_lst = CBFL_training_clusters(args=args, model_in_path_root=args.model_in_path+"_FLASR", model_out_path=args.model_out_path,
                                    train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
                                    test_dataset=test_dataset, Nth_Kmeans_update=Nth_Kmeans_update, init_weights_lst=global_weights_lst)
        # update global model for each cluster
        for j in range(args.num_lms):
            global_weights = global_weights_lst[j]
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            if args.STAGE == 1: # freeze all, train ASR decoder alone
                save_weights(folder_to_save, global_weights)
            else:
                model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final/", target_weight=global_weights, network="ASR")
                                                                                            # update decoder
                model.save_pretrained(folder_to_save + "/final")

        if i < (int(args.epochs / args.N_Kmeans_update) - 1):                               # update cluster except for the last round
            model_in_path = args.model_in_path
            args.model_in_path = args.model_out_path+"#_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1))
            Kmeans_clustering(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
                                                                                            # re-cluster
            # re-load dataset
            train_dataset_supervised, train_dataset_unsupervised, test_dataset = get_dataset(args)
            args.model_in_path = model_in_path
            for j in range(args.num_lms):
                shutil.rmtree(args.model_out_path+"_cluster" + str(j) + "_CBFLASR_global_round" + str(int((i+1)*args.N_Kmeans_update - 1)))
                                                                                                # remove aggregated results
        

    
    #############
    # 後面繼續

def get_dataset(args):                                                                      # return train_dataset_supervised, train_dataset_unsupervised, test_dataset
    args.dataset = "adress"                                                                 # get supervised dataset (adress)
    train_dataset_supervised, test_dataset = get_raw_dataset(args)                          # get dataset
    args.dataset = "adresso"                                                                # get unsupervised dataset (adresso)
    train_dataset_unsupervised, _ = get_raw_dataset(args)                                   # get dataset w.o. testing set

    if args.FL_STAGE == 4:                                                                  # new dataset loaded
        return train_dataset_supervised, train_dataset_unsupervised, test_dataset           # simply return them
    
    if train_dataset_unsupervised != None:                                                  # if train_dataset_unsupervised needed, add transcripts
        # 創建一個 TranscriptDataset 物件
        TSL = TeacherStudentLearning()
        out_root='/mnt/Internal/FedASR/Data'
        out_dirname='transcript_whisper'
        out_filename='transcript_train.csv'
        out_path=f"{out_root}/{out_dirname}"
        in_file=f"{out_root}/{out_dirname}/{out_filename}"
        with open('/home/FedASR/dacs/federated/src/transcript.json') as f:
            transcript = json.load(f)                                                       # read whisper-generated transcript
        
        # remove punctuation
        transcript_norm = []
        for text in transcript:
            pred_text=TextNor(text.upper().strip())
            transcript_norm.append(pred_text)

        train_dataset_addresso=TSL.add_transcript_to_dataset(train_dataset_unsupervised, transcript_norm)
        train_dataset_addresso_validated=TSL.FilterAvailAudios(train_dataset_addresso)      # remove empty files

        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        train_dataset_addresso_validated = train_dataset_addresso_validated.map(lambda x: prepare_dataset(x, processor=processor, with_transcript=True), num_proc=10)

        train_dataset_unsupervised=train_dataset_addresso_validated

    return train_dataset_supervised, train_dataset_unsupervised, test_dataset

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    
    args = args_parser()                                                                    # get configuration
    exp_details(args)                                                                       # print out details based on configuration
    
    # 1868 supervised samples + 3033 unsupervised samples = 4901 samples in total
    train_dataset_supervised, train_dataset_unsupervised, test_dataset = get_dataset(args)
    ##
    # debug 區
    ##
    #evaluateASR(args, 9, test_dataset, train_dataset_supervised)
    #aaa=ccc
    #args.STAGE = 1 
    #stage4_debug(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
    #aaa=ccc
    #train_dataset_unsupervised = reorder_col(train_dataset_supervised, train_dataset_unsupervised) # reorder as target_dataset

    #stored = "./dataset/ADReSSo_clustered/" + "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/train_ADReSSo.csv".split("/")[-1].split(".")[0]
    #train_dataset_unsupervised.save_to_disk(stored)                                                            # save for later use
    #client_train(args, args.model_in_path+"_FLASR", args.model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, 0,
     #                                                       0, 0, None)
    #FL_training_rounds(args=args, model_in_path_root=args.model_in_path+"_FLASR", model_out_path=args.model_out_path,
    #                                    train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
    #                                    test_dataset=test_dataset, cluster_id=0)
    #print(train_dataset_unsupervised.column_names)
    #aaa=ccc

    if args.EXTRACT != True:                                                                # Training
        if args.FL_STAGE == 1:                                                              # global train ASR
            print("| Start FL Training Stage 1 -- Global Train ASR |")
            args.STAGE = 0                                                                  # 0: train ASR encoder & decoder
            GlobalTrainASR(args=args, train_dataset_supervised=train_dataset_supervised, test_dataset=test_dataset)                      
                                                                                            # Train ASR encoder & decoder
            print("| FL Training Stage 1 Done|")
        elif args.FL_STAGE == 2:                                                            # FL train ASR for 1 round
            print("| Start FL Training Stage 2|")
            args.STAGE = 0                                                                  # train ASR encoder & decoder
            FL_TrainASR(args=args, train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised, 
                        test_dataset=test_dataset)
            print("| FL Training Stage 2 Done|")
        elif args.FL_STAGE == 3:                                                            # K-means clustering
            print("| Start FL Training Stage 3|")
            Kmeans_clustering(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
            print("| FL Training Stage 3 Done|")

            # 繼續stage 4
            print("| Start FL Training Stage 4|")
            args.STAGE = 0                                                                  # train ASR encoder as well
            #args.STAGE = 1                                                                  # train ASR decoder
            args.FL_STAGE = 4
            args.model_in_path = "/mnt/External/Seagate/weitung/save/data2vec-audio-large-960h"
            train_dataset_supervised, train_dataset_unsupervised, test_dataset = get_dataset(args)
            FL_TrainASRdecoder(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
            print("| FL Training Stage 4 Done|")

        elif args.FL_STAGE == 4:                                                            # FL train ASR decoder
            print("| Start FL Training Stage 4|")
            args.STAGE = 0                                                                  # train ASR encoder as well
            #args.STAGE = 1                                                                  # train ASR decoder
            FL_TrainASRdecoder(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
            #stage4_debug(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset)
            print("| FL Training Stage 4 Done|")
        else:
            print("Only FL Training Stage 1-4 is available, current FL_STAGE = ", args.FL_STAGE)

    else:
        #getTestsetEmb(args, args.model_in_path, test_dataset, save_csv=True)
        #get_overall_wer(args, test_dataset)
        if args.eval_mode == 2:                                                                         # 測在local test上
            _, test_dataset_clients = train_split_supervised(args, train_dataset_supervised, client_id=0, cluster_id=None)
                                                                                                        # client 0的test set
            # 單一client的結果
            """
            if len(test_dataset_clients) == 0:                                                          # no testing sample
                get_overall_wer(args, test_dataset)                                                     # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_clients, LOCAL=True)                                 # get WER for client's testing set

            """
            for i in range(args.num_users-1):                                                                        # for all clients that perform training
                # filter出client i的local_test_dataset，不分群（但dataset有cluster_id）
                # supervised才有人工的ground truth label
                _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i+1, cluster_id=None)
                test_dataset_clients = concatenate_ds(test_dataset_clients, test_dataset_client)        # add to ovarall test set of clients

                # 單一client的結果
                """
                if len(test_dataset_client) == 0:                                                       # no testing sample for this client
                    get_overall_wer(args, test_dataset)                                                 # get WER for global testing set
                else:
                    get_overall_wer(args, test_dataset_client, LOCAL=True)                              # get WER for client's testing set
                """
            # client test set合起來的結果
            if len(test_dataset_clients) == 0:                                                          # no testing sample
                get_overall_wer(args, test_dataset)                                                     # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_clients, LOCAL=True)                                 # get WER for clients' testing set
        # global的結果
        get_overall_wer(args, test_dataset)                                                     # get WER for global testing set
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
