#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

from transformers import Wav2Vec2Processor
from datasets import Dataset
import librosa
import numpy as np
import pandas as pd
import os
from datasets import load_from_disk
import scipy
import argparse
import re
from datasets import * 
from transformers import Data2VecAudioConfig
from models import Data2VecAudioForCTC_CPFL
from jiwer import wer

CPFL_codeRoot = os.environ.get('CPFL_codeRoot')
CPFL_dataRoot = os.environ.get('CPFL_dataRoot')



# some parameters
parser = argparse.ArgumentParser()
#parser.add_argument('-model', '--model_path', type=str, default="./saves/data2vec-audio-large-960h", help="Where the model is saved")
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read smping rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--AudioLoadFunc', default='librosa', help="scipy function might perform faster")
args = parser.parse_args(args=[])


def prepare_dataset(batch, processor, with_transcript=True):
    if "input_values" not in batch.keys():                                  # get input_values only for the 1st time
        audio = batch["array"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        
    if with_transcript:                                                     # if given transcript
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids            # generate labels
        
    return batch

def ID2Label(ID, spk2label):
    name = ID.split("_")                                                    # from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is healthy (not AD)
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return dementia label for this file

def csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
                csv_path = '{}/mid_csv/test.csv'.format(args.root_dir),
                dataset_path = "./dataset/", with_transcript=True):
    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(csv_path)                                            # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    if with_transcript:
        my_dict["text"] = []                                                # ground truth transcript if given
    my_dict["dementia_labels"] = []

    if with_transcript:                                                     # ADReSS
        spk2label=np.load(path2_ADReSS_dict, allow_pickle=True).tolist()
    else:                                                                   # ADReSSo
        spk2label=np.load(path2_ADReSSo_dict, allow_pickle=True).tolist()

    i = 1
    for file_path in dataset['path']:                                       # for all files
        if 'sentence' in dataset.features:                                  # if col "sentence" exists
            if dataset['sentence'][i-1] == None:                            # but no info
                continue                                                    # skip to next file
        if args.AudioLoadFunc == 'librosa':
            try:
                sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  
                                                                            # read audio w/ 16k sr
            except   ValueError:                                            # skip files that can't be loaded                                                 
                print("Err file = ", audio_path,file_path)
        else:
            s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
            sig=librosa.util.normalize(sig)
        if len(sig) > 1600:                                                 # get rid of audio that's too short
            my_dict["path"].append(file_path)                               # add path
            my_dict["array"].append(sig)                                    # add audio wave
            if with_transcript:
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
            my_dict["dementia_labels"].append(ID2Label(ID=file_path, spk2label=spk2label))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset

def get_raw_dataset(args):                                                  # return whole training & testing set of ADReSS
    if args.dataset == 'adress':                                            # for ADReSS dataset
        if args.FL_STAGE == 4:
            dataset_path = args.dataset_path_root + "/ADReSS_clustered/"    # load clustered dataset
        else:
            dataset_path = args.dataset_path_root + "/"                     # load dataset w.o. cluster info

        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        
        # load and map train data
        train_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/mid_csv/train.csv", dataset_path=dataset_path)
        train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)

        # load and map test data
        test_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/mid_csv/test.csv", dataset_path=dataset_path)
        test_dataset = test_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)

    return train_dataset, test_dataset

def reorder_col(datasetA, datasetB):                                        # order B as A, return re-ordered B
    # turn target Dataset to dataframe
    dfB = datasetB.to_pandas()

    # order B as A
    column_order = datasetA.column_names                                    # A's col order
    dfB = dfB[column_order]

    datasetB_reordered = Dataset.from_pandas(dfB)                           # turn back to type 'Dataset'
    return datasetB_reordered

def average_weights(w, num_training_samples_lst, WeightedAvg):              # given list of clients' weights
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])                                             # save 1st client's model weight
    if WeightedAvg:                                                         # taking weighted sum
        print("Perform weighted Avg on models!!")
        for key in w_avg.keys():                                            # each layer
            w_avg[key] = w[0][key]*num_training_samples_lst[0]              # for 1st client
            for i in range(1, len(w)):                                      # for each participated client
                w_avg[key] += w[i][key]*num_training_samples_lst[i]         # for weighted sum
            w_avg[key] = torch.div(w_avg[key], np.array(num_training_samples_lst).sum()) 
                                                                            # weighted sum
    else:
        for key in w_avg.keys():                                            # each layer
            for i in range(1, len(w)):                                      # for each participated client
                w_avg[key] += w[i][key]                                     # sum up weight for this layer
            w_avg[key] = torch.div(w_avg[key], len(w))                      # take average (element-wise divide)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Current Stage   : {args.FL_STAGE}\n')

    print('    Federated parameters:')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Eval step is set to  : {args.eval_steps}')
    print(f'    Current training type: {args.training_type}')
    print(f'    Current number of clusters: {args.num_lms}')

    return

def add_cluster_id(example, cluster_id):
    example["cluster_id"] = cluster_id
    return example

def gen_mapping_fn(args, processor, model_lst):
    def map_to_result(batch):                                               # 1 sample per batch
        with torch.no_grad():
            if args.num_lms > 1:                                            # for multi-cluster
                model_id = batch["cluster_id"]                              # get cluster_id for this sample
                model = model_lst[model_id]                                 # use corresponding model
            else:
                model = model_lst[0]                                        # use the 1st model for uni-cluster
            # decode using corresponding model
            input_values = torch.tensor(batch["input_values"]).unsqueeze(0).to("cuda")
            model = model.to("cuda")
            logits = model(input_values).logits
            # save result
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids)[0]
            batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
        return batch
    return map_to_result

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import Levenshtein
from jiwer import transforms as tr
from jiwer.transformations import wer_default, wer_standardize, cer_default_transform
from detail_wer import _preprocess

def _get_operation_counts(
    source_string: str, destination_string: str
) -> Tuple[int, int, int, int]:
    """
    Check how many edit operations (delete, insert, replace) are required to
    transform the source string into the destination string. The number of hits
    can be given by subtracting the number of deletes and substitutions from the
    total length of the source string.
    :param source_string: the source string to transform into the destination string
    :param destination_string: the destination to transform the source string into
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """
    editops = Levenshtein.editops(source_string, destination_string)
            
    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)
    
    return hits, substitutions, deletions, insertions

def compute_measures(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate error measures between a set of ground-truth sentences and a set of
    hypothesis sentences.
    The set of sentences can be given as a string or a list of strings. A string
    input is assumed to be a single sentence. A list of strings is assumed to be
    multiple sentences which need to be evaluated independently. Each word in a
    sentence is separated by one or more spaces. A sentence is not expected to end
    with a specific token (such as a `.`). If the ASR system does delimit sentences
    it is expected that these tokens are filtered out.
    The optional `transforms` arguments can be used to apply pre-processing to
    respectively the ground truth and hypotheses input. By default, the following
    transform is applied to both the ground truth and hypothesis string(s). These
    steps are required and necessary in order to compute the measures.
    1) The start and end of a string are stripped of white-space symbols
    2) Contiguous spaces (e.g `   `) are reduced to a single space (e.g ` `)
    3) A sentence (with a single space (` `) between words) is reduced to a
       list of words
    Any non-default transformation is required to reduce the input to at least
    one list of words in order to facility the computation of the edit distance.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: a dict with WER, MER, WIP and WIL measures as floating point numbers
    """
    # deprecated old API
    if "standardize" in kwargs:
        warnings.warn(
            UserWarning(
                "keyword argument `standardize` is deprecated. "
                "Please use `truth_transform=jiwer.transformations.wer_standardize` and"
                " `hypothesis_transform=jiwer.transformations.wer_standardize` instead"
            )
        )
        truth_transform = wer_standardize
        hypothesis_transform = wer_standardize
    if "words_to_filter" in kwargs:
        warnings.warn(
            UserWarning(
                "keyword argument `words_to_filter` is deprecated. "
                "Please compose your own transform with `jiwer.transforms.RemoveSpecificWords"
            )
        )
        t = tr.RemoveSpecificWords(kwargs["words_to_filter"])
        truth = t(truth)
        hypothesis = t(hypothesis)

    # validate input type
    if isinstance(truth, str):
        truth = [truth]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in truth):
        raise ValueError("one or more groundtruths are empty strings")

    # Preprocess truth and hypothesis
    trans = truth
    pred = hypothesis

    truth, hypothesis = _preprocess(
        truth, hypothesis, truth_transform, hypothesis_transform
    )

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    H, S, D, I = 0, 0, 0, 0

    # also keep track of the total number of ground truth words and hypothesis words
    gt_tokens, hp_tokens = 0, 0
    
    i = 0
    for groundtruth_sentence, hypothesis_sentence in zip(truth, hypothesis):
        # Get the operation counts (#hits, #substitutions, #deletions, #insertions)       
        hits, substitutions, deletions, insertions = _get_operation_counts(
            groundtruth_sentence, hypothesis_sentence
        )

            
        H += hits
        S += substitutions
        D += deletions
        I += insertions
        gt_tokens += len(groundtruth_sentence)
        hp_tokens += len(hypothesis_sentence)
        i = i + 1

    # Compute Word Error Rate
    wer = float(S + D + I) / float(H + S + D)

    # Compute Match Error Rate
    mer = float(S + D + I) / float(H + S + D + I)

    # Compute Word Information Preserved
    wip = (float(H) / gt_tokens) * (float(H) / hp_tokens) if hp_tokens >= 1 else 0

    # Compute Word Information Lost
    wil = 1 - wip        

    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "hits": H,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
    }

def record_WER(args, result, cluster_num, test_data="global"):
    wer_result = compute_measures(truth=result["text"], hypothesis=result["pred_str"])
    # filter out AD and HC
    HC_result = result.filter(lambda example: example["dementia_labels"]==0 and example['text'] != '')
    AD_result = result.filter(lambda example: example["dementia_labels"]==1 and example['text'] != '')

    if len(HC_result["text"]) != 0:                                         # if sample exists, compute wer
        wer_HC = compute_measures(truth=HC_result["text"], hypothesis=HC_result["pred_str"])
    else:
        wer_HC = {"wer": "No sample"}                                       # or record "No sample"

    if len(AD_result["text"]) != 0:                                         # if sample exists, compute wer
        wer_AD = compute_measures(truth=AD_result["text"], hypothesis=AD_result["pred_str"])
    else:
        wer_AD = {"wer": "No sample"}                                       # or record "No sample"

    if cluster_num != None:                                                 # record cluster_id if given
        model_name = args.model_in_path.split("/")[-1] + "_cluster" + str(cluster_num)
    else:
        model_name = args.model_in_path.split("/")[-1]

    model_name = model_name + "_" + test_data

    data = {
    'model': model_name,
    'WER': [wer_result["wer"]],
    'AD_WER': [wer_AD["wer"]],
    'HC_wer': [wer_HC["wer"]],
    'HITS': [wer_result["hits"]],
    'substitutions': [wer_result["substitutions"]],
    'deletions': [wer_result["deletions"]],
    'insertions': [wer_result["insertions"]]
    }
    df = pd.DataFrame(data)

    # check if file exists
    file_exists = os.path.isfile('./results/Overall_WER.csv')

    # if file exists, no header
    if file_exists:
        df.to_csv('./results/Overall_WER.csv', mode='a', header=False, index=False)
    else:
        # create new file
        df.to_csv('./results/Overall_WER.csv', index=False)

def get_overall_wer(args, dataset, test_data="global"):
    torch.set_num_threads(1)
    # load ASR model
    mask_time_prob = 0                                                      # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)

    model_lst = []
    if args.num_lms > 1:                                                    # multi-cluster
        for cluster_id in range(args.num_lms):                              # load model 1 by 1
            txt = args.model_in_path.split("#")
            model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
            model_lst.append(model)
    else:                                                                   # load from args.model_in_path
        model = load_model(args, args.model_in_path, config)
        model_lst.append(model)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

    result = dataset.map(gen_mapping_fn(args, processor, model_lst))
    record_WER(args, result, None, test_data=test_data)


def load_model(args, model_in_path, config):                                # model_in_path w.o. "/final/"
    file_to_check = model_in_path + "/decoder_weights.pth"
    if os.path.isfile(file_to_check):
        # file exists
        model = Data2VecAudioForCTC_CPFL.from_pretrained(args.model_out_path[:-25] + "data2vec-audio-large-960h_FLASR_global/final", config=config, args=args)
        # load decoder's weight
        decoder_state_dict = torch.load(model_in_path + "/decoder_weights.pth")
        model.lm_head.load_state_dict(decoder_state_dict)
    else:
        model = Data2VecAudioForCTC_CPFL.from_pretrained(model_in_path+"/final/", config=config, args=args)
    
    return model

############################################################################################
# Splits-related: client, train / test
############################################################################################
# for each INV & PAR, take 80% of data as training and 20%  as testing
def split_train_test_spk(source_dataset, client_spk, identity, DEV):
    # identity: "INV" or "PAR"
    subsetA = source_dataset.filter(lambda example: example["path"].startswith(client_spk+"_"+identity))
                                                    # filter out a single spk
    subsetA = subsetA.sort("path")                  
    LEN_subsetA = len(subsetA)                      # num of sample for this spk
    if DEV:
        num_sample_train = max(1, int(LEN_subsetA*0.7))       # min 1, use 70% of samples as training
        num_sample_trainDev = max(1, int(LEN_subsetA*0.8))    # min 1, use 80% as (training + dev)
        
        if num_sample_train == 0:                             # if 0 sample
            train_dataset = Dataset.from_dict({})             # return empty dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample_train))
                                                              # select 70% as training
        if num_sample_train == num_sample_trainDev:           # if 0 sample
            test_dataset = Dataset.from_dict({})              # return empty dataset
        else:        
            test_dataset = subsetA.select(range(num_sample_train, num_sample_trainDev))
                                                              # select 10% as dev
    else:
        num_sample = max(1, int(LEN_subsetA*0.8))             # min 1, use 80% as training
        
        if num_sample == 0:                                   # if 0 sample
            train_dataset = Dataset.from_dict({})             # return empty dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample))
                                                              # select 80% as training
        if num_sample == LEN_subsetA:                         # if 0 sample
            test_dataset = Dataset.from_dict({})              # return empty dataset
        else:        
            test_dataset = subsetA.select(range(num_sample, LEN_subsetA))
                                                              # select 20% as testing
    return train_dataset, test_dataset

from datasets import concatenate_datasets
def concatenate_ds(datasetA, datasetB):
    if len(datasetA) != 0 and len(datasetB) != 0:             # if both non-empty, combine them
        concatenated_dataset = concatenate_datasets([datasetA, datasetB])
        return concatenated_dataset
    
    # at least one of them is empty
    if len(datasetA) != 0:                                    # A not empty, return it
        return datasetA    
    return datasetB                                           # return B

# return train / test set of this client
def split_train_test_client(client_spks, source_dataset, DEV=False):    
                                                              # default: no dev
    # for 1st spk_id, get training(80%) and testing(20%) data for INV and PAR
    client_spk = client_spks[0]
    train_dataset_INV, test_dataset_INV = split_train_test_spk(source_dataset, client_spk, "INV", DEV)
    train_dataset_PAR, test_dataset_PAR = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)

    # combine INV and PAR
    train_dataset_client = concatenate_ds(train_dataset_INV, train_dataset_PAR)
    test_dataset_client = concatenate_ds(test_dataset_INV, test_dataset_PAR)

    for i in range(len(client_spks)-1):                       # for each spk_id
        # get training(80%) and testing(20%) data for INV and PAR
        client_spk = client_spks[i+1]
        train_dataset_INV, test_dataset_INV = split_train_test_spk(source_dataset, client_spk, "INV", DEV)
        train_dataset_PAR, test_dataset_PAR = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)

        # combine INV and PAR
        train_dataset_spk = concatenate_ds(train_dataset_INV, train_dataset_PAR)
        test_dataset_spk = concatenate_ds(test_dataset_INV, test_dataset_PAR)
        #print(len(train_dataset_spk), len(test_dataset_spk))

        # combine to client data
        train_dataset_client = concatenate_ds(train_dataset_client, train_dataset_spk)
        test_dataset_client = concatenate_ds(test_dataset_client, test_dataset_spk)    

    return train_dataset_client, test_dataset_client

# use when each spk is a client
def client2spk(client_id):
  client2spk_dict = { '1': 'S058',  '2': 'S030',  '3': 'S064',  '4': 'S104',  '5': 'S048', 
                      '6': 'S118',  '7': 'S122',  '8': 'S001',  '9': 'S087', '10': 'S013',
                     '11': 'S025', '12': 'S083', '13': 'S067', '14': 'S068', '15': 'S111', 
                     '16': 'S028', '17': 'S015', '18': 'S108', '19': 'S095', '20': 'S002', 
                     '21': 'S072', '22': 'S020', '23': 'S148', '24': 'S144', '25': 'S110', 
                     '26': 'S124', '27': 'S129', '28': 'S071', '29': 'S136', '30': 'S140', 
                     '31': 'S145', '32': 'S032', '33': 'S101', '34': 'S103', '35': 'S139', 
                     '36': 'S038', '37': 'S153', '38': 'S035', '39': 'S011', '40': 'S132', 
                     '41': 'S006', '42': 'S149', '43': 'S041', '44': 'S079', '45': 'S107', 
                     '46': 'S063', '47': 'S061', '48': 'S125', '49': 'S062', '50': 'S012', 
                     '51': 'S138', '52': 'S024', '53': 'S052', '54': 'S142'}
  return [client2spk_dict[str(client_id+1)]]
    
# Mode 1: no client test
# Mode 2: client test by utt
# Mode 2: client dev by utt
def train_split_supervised(args, dataset, client_id, cluster_id):
    # generate sub- training set for given user-ID
    if args.num_users > 5:                                                                 # for "spk as client" setting
        client_spks = client2spk(client_id)
        print("Current spk: ", client_spks)
    elif client_id == "public":                                                            # get spk_id for public dataset, 54 PAR (50% of all training set)
        client_spks = ['S086', 'S021', 'S018', 'S156', 'S016', 'S077', 'S027', 'S116', 'S143', 'S082', 'S039', 'S150', 'S004', 'S126', 'S137', 
        'S097', 'S128', 'S059', 'S096', 'S081', 'S135', 'S094', 'S070', 'S049', 'S080', 'S040', 'S076', 'S093', 'S141', 'S034', 'S056', 'S090', 
        'S130', 'S092', 'S055', 'S019', 'S154', 'S017', 'S114', 'S100', 'S036', 'S029', 'S127', 'S073', 'S089', 'S051', 'S005', 'S151', 'S003', 
        'S033', 'S007', 'S084', 'S043', 'S009']                                             # 27 AD + 27 HC
    elif client_id == "public2":                                                            # get spk_id for public dataset, 54 PAR (50% of all training set) from clients
        client_spks = ['S058', 'S030', 'S064', 'S104', 'S048', 'S118', 'S122', 'S001', 'S087', 'S013', 'S025', 'S083', 'S067', 'S068', 'S111', 
        'S028', 'S015', 'S108', 'S095', 'S002', 'S072', 'S020', 'S148', 'S144', 'S110', 'S124', 'S129', 'S071', 'S136', 'S140', 'S145', 'S032', 
        'S101', 'S103', 'S139', 'S038', 'S153', 'S035', 'S011', 'S132', 'S006', 'S149', 'S041', 'S079', 'S107', 'S063', 'S061', 'S125', 'S062', 
        'S012', 'S138', 'S024', 'S052', 'S142']                                             # 27 AD + 27 HC
        print("Train with all client data")
    elif client_id == 0:                                                                    # 10 PAR w/ 10 AD
        client_spks = ['S139', 'S125', 'S145', 'S149', 'S138', 'S144', 'S101', 'S136', 'S148', 'S108']
        # random
        #client_spks = ['S087', 'S041', 'S079', 'S063', 'S062', 'S138']
    elif client_id == 1:                                                                    # 12 PAR w/ 9 AD
        client_spks = ['S030', 'S124', 'S013', 'S111', 'S140', 'S095', 'S104', 'S006', 'S087', 'S153', 'S107', 'S142']
        # random
        #client_spks = ['S048', 'S122', 'S015', 'S002', 'S072', 'S020', 'S148', 'S071', 'S136', 'S139', 'S011', 'S125', 'S052', 'S142']
    elif client_id == 2:                                                                    # 10 PAR w/ 5 AD
        client_spks = ['S110', 'S028', 'S083', 'S038', 'S079', 'S067', 'S129', 'S052', 'S024', 'S132']
        # random
        #client_spks = ['S030', 'S001', 'S028', 'S108', 'S110', 'S124', 'S129', 'S103', 'S149', 'S107', 'S061', 'S024']
    elif client_id == 3:                                                                    # 12 PAR w/ 3 AD
        client_spks = ['S071', 'S012', 'S032', 'S103', 'S122', 'S118', 'S020', 'S015', 'S002', 'S041', 'S062', 'S072']
        # random
        #client_spks = ['S104', 'S025', 'S111', 'S144', 'S145', 'S032', 'S038', 'S153', 'S035', 'S132']
    elif client_id == 4:                                                                    # 10 PAR w/ 10 HC
        client_spks = ['S011', 'S025', 'S058', 'S001', 'S048', 'S064', 'S068', 'S063', 'S061', 'S035']                                    
        # random
        #client_spks = ['S058', 'S064', 'S118', 'S013', 'S083', 'S067', 'S068', 'S095', 'S140', 'S101', 'S006', 'S012']
    else:
        print("Train with whole dataset!!")
        return dataset

    print("Generating client training set for client ", str(client_id), "...")
    if cluster_id == None:                                                                  # no cluster
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
    else:                                                                                   # get cluster-specific dataset
        print("Generating client training set for cluster ", str(cluster_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)) and example['cluster_id'] == cluster_id)
    
    # Mode 1: no client test
    if args.eval_mode == 1:                             
        return client_train_dataset, None
    # Mode 2: client test by utt
    elif args.eval_mode == 2: 
        train_dataset_client, test_dataset_client = split_train_test_client(client_spks, client_train_dataset)
        return train_dataset_client, test_dataset_client
    elif args.eval_mode == 3:                                                               # 70% training & 10% dev
        train_dataset_client, dev_dataset_client = split_train_test_client(client_spks, client_train_dataset, DEV=True)
        return train_dataset_client, dev_dataset_client
    # default: no client test
    return client_train_dataset, None 
    
def evaluateASR(args, global_round, global_test_dataset, train_dataset_supervised=None):
    if args.chosen_clients == True:                                                                 # train only the chosen clients
        idxs_users = [0, 4]
    else:
        idxs_users = range(args.num_users)                                                          # train w/ all clients

    # eval client model   
    for i in idxs_users:                                                                            # for all clients that perform training
        save_path = args.model_out_path + "_client" + str(i) + "_round" + str(global_round)
        if args.num_lms > 1:                                                                        # more than 1 cluster
            save_path += "#"

        if args.training_type == 1:                                                                 # supervised
            save_path += "_Training" + "Address"

        if args.FL_type == 3:                                                                       # FML
            save_path += "_localModel"                                                              # eval local model
        args.model_in_path = save_path
        
        get_overall_wer(args, global_test_dataset)                                                  # get WER for global testing set
        
        # Mode 2: client test by utt
        # Mode 3: client dev by utt
        if args.eval_mode == 2 or args.eval_mode == 3:                                              # test on 20%
            origin_eval_mode = args.eval_mode                                                       # record eval_mode for later use
            args.eval_mode = 2                                                                      # test on 20%
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="test")                        # get WER for each client's testing set
            args.eval_mode = origin_eval_mode                                                       # back to original eval_mode
        
        if args.eval_mode == 3:                                                                     # test on 10% dev
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="dev")                         # get WER for each client's testing set
    
    # eval aggregated model    
    if args.num_lms > 1:                                                                            # more than 1 cluster
        args.model_in_path = args.model_out_path+"#_CPFLASR_global_round" + str(global_round)
    else:                                                                                           # if 1 cluster, evaluate that 1
        args.model_in_path = args.model_out_path+"_cluster0_CPFLASR_global_round" + str(global_round)
    
    get_overall_wer(args, global_test_dataset)                                                      # get WER for global testing set
    
    # Mode 2: client test by utt
    # Mode 3: client dev by utt
    if args.eval_mode == 2 or args.eval_mode == 3:                                                  # test on local test & dev
        for i in idxs_users:                                                                        # for all clients that perform training
            origin_eval_mode = args.eval_mode                                                       # record eval_mode for later use
            args.eval_mode = 2
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="test")                        # get WER for each client's testing set
            args.eval_mode = origin_eval_mode                                                       # back to original eval_mode
    
    if args.eval_mode == 3:                                                                         # test on 10% dev
        for i in idxs_users:  
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="dev")                         # get WER for each client's testing set
        
