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



# 一些參數
parser = argparse.ArgumentParser()
#parser.add_argument('-model', '--model_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5", help="Where the model is saved")
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read smping rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--AudioLoadFunc', default='librosa', help="用scipy function好像可以比較快")
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
    name = ID.split("_")                                                    #  from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is CC
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return dementia label for this file

# 改版的 
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
        if 'sentence' in dataset.features:                                  # 只有有sentence且sentence那格沒資訊的時候才跳過
            if dataset['sentence'][i-1] == None:
                continue
        if args.AudioLoadFunc == 'librosa':
            try:
                sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  
                                                                            # read audio w/ 16k sr
            except   ValueError:                                            # 跳過讀不進來的音檔                                                 
                print("Err file = ", audio_path,file_path)
        else:
            s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
            sig=librosa.util.normalize(sig)
        if len(sig) > 1600:                                                 # get rid of audio that's too short
            my_dict["path"].append(file_path)                               # add path
            my_dict["array"].append(sig)                                    # add audio wave
            if with_transcript:
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
            my_dict["dementia_labels"].append(ID2Label(ID=file_path,
                                                        spk2label=spk2label))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset

def TextNor(text):                                                          # text normalization on given text
    # get rid of things inside []
    start = text.split("[")                                                 # split by [
    result = ''
    for i in range(len(start)):                                             # for all parts
        end = start[i].split("]")
        if (len(end) > 1):
            result += end[1]                                                # after [ & ]
        else:
            result += end[0]                                                # before [

    # get rid of &=xxx
    start = result.split(" ")
    result = ''
    for i in range(len(start)):
        if start[i][:2] != '&=':                                            # skip &=xxx
            result += start[i] + ' '                                        # add space back

    # get rid of ‡
    start = result.split("‡")
    result = ''
    for i in range(len(start)):
        result += start[i] + ' '                                            # add space back


    # punctuation-free
    result = result.replace("_", " ").replace("xxx", " ").replace("www", " ").replace("??", "")
                                                                            # xxx for Unintelligible words
                                                                            # www for Untranscribed Material
    punc = '"~!@#$%^&*()_+=><,./?:;{}[]|-‡'
    out = re.sub(r"[%s]+" %punc, "", result)

    out = " ".join(out.split())                                             # get rid of repeated spaces
    return out

def get_raw_dataset(args):                                                  # return whole training & testing set of ADReSS or ADReSSo
    if args.dataset == 'adress':                                            # for ADReSS dataset
        if args.FL_STAGE == 4:
            dataset_path = args.dataset_path_root + "/ADReSS_clustered/"
            #dataset_path = "./dataset/ADReSS_clustered/"                    # load clustered dataset
        else:
            dataset_path = args.dataset_path_root + "/"

        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

        if args.training_type == 2:                                         # 2 for semi-supervised only
            train_dataset = None
            print("Train without ADReSS dataset.")
        else:
            # load and map train data
            train_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/mid_csv/train.csv", dataset_path=dataset_path)
            #train_data = train_data.map(prepare_dataset, num_proc=10)
            train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)
        # load and map dev data
        #dev_data = csv2dataset(path = f"{CPFL_dataRoot}/mid_csv/dev.csv")
        #dev_data = dev_data.map(prepare_dataset, num_proc=10)

        # load and map test data
        test_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/mid_csv/test.csv", dataset_path=dataset_path)
        #test_data = test_data.map(prepare_dataset, num_proc=10)
        test_dataset = test_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)

    elif args.dataset == 'adresso':                                         # for ADReSSo dataset
        if args.FL_STAGE == 4:
            dataset_path = args.dataset_path_root + "/ADReSSo_clustered/"
            #dataset_path = "./dataset/ADReSSo_clustered/"                   # load clustered dataset
        else:
            dataset_path = args.dataset_path_root + "/ADReSSo/"

        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

        if args.training_type == 1:                                         # 1 for supervised only
            train_dataset = None
            print("Train without ADReSSo dataset.")
        else:
            # load train data
            train_data = csv2dataset(audio_path = "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/clips/",
                                    csv_path = "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/train_ADReSSo.csv",
                                    dataset_path=dataset_path, with_transcript=False)
            # map to desired form
            train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor, with_transcript=False), num_proc=10)
        test_dataset=None                                                   # No testing set for ADReSSo

    return train_dataset, test_dataset

def reorder_col(datasetA, datasetB):                                        # order B as A, return re-ordered B
    # turn target Dataset to dataframe
    dfB = datasetB.to_pandas()

    # order B as A
    column_order = datasetA.column_names                                    # A's col order
    dfB = dfB[column_order]

    datasetB_reordered = Dataset.from_pandas(dfB)                           # turn back to type 'Dataset'
    return datasetB_reordered

def average_weights(w, num_training_samples_lst, WeightedAvg):                           # given list of clients' weights
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])                                             # save 1st client's model weight
    if WeightedAvg:                                                         # taking weighted sum
        print("Perform weighted Avg on models!!")
        for key in w_avg.keys():                                                # each layer
            w_avg[key] = w[0][key]*num_training_samples_lst[0]                  # for weighted sum
            for i in range(1, len(w)):                                          # for each participated client
                w_avg[key] += w[i][key]*num_training_samples_lst[i]             # for weighted sum
            w_avg[key] = torch.div(w_avg[key], np.array(num_training_samples_lst).sum()) # weighted sum
    else:
        for key in w_avg.keys():                                                # each layer
            for i in range(1, len(w)):                                          # for each participated client
                w_avg[key] += w[i][key]                                         # sum up weight for this layer
            w_avg[key] = torch.div(w_avg[key], len(w))                          # take average (element-wise divide)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Current Stage   : {args.FL_STAGE}\n')

    print('    Federated parameters:')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    eval step is set to  : {args.eval_steps}')
    print(f'    Current training type: {args.training_type}')
    print(f'    Current number of clusters: {args.num_lms}')

    return

#def add_cluster_id(example, cluster_id):
#    example["cluster_id"] = cluster_id
#    return example

def add_cluster_id(example, cluster_id):
    #for example, cluster_id in zip(batch, cluster_ids):
    example["cluster_id"] = cluster_id
    return example

def add_entropy(example, entropy, first_time):
  if first_time:
    example["entropy_history"] = [entropy]                                  # assign 1st value as list
  else:
    example["entropy_history"] = list(np.concatenate((example["entropy_history"][0], [entropy[-1]]), axis=0))
                                                                            # extend new one into existing features, size = [1, 舊維度+1]
  return example

def gen_mapping_fn(args, processor, model_lst):
    def map_to_result(batch):                                               # 一個batch只有一個sample
        with torch.no_grad():
            if args.num_lms > 1:                                            # multi-cluster
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
    # 這個set的AD and HC
    HC_result = result.filter(lambda example: example["dementia_labels"]==0 and example['text'] != '')
    #print(HC_result["dementia_labels"]) # 有確實filter
    
    AD_result = result.filter(lambda example: example["dementia_labels"]==1 and example['text'] != '')
    #print(AD_result["dementia_labels"]) # 有確實filter

    if len(HC_result["text"]) != 0:                                         # 有sample就算WER
        wer_HC = compute_measures(truth=HC_result["text"], hypothesis=HC_result["pred_str"])
    else:
        wer_HC = {"wer": "No sample"}                                       # 沒有sample紀錄沒有

    if len(AD_result["text"]) != 0:                                         # 有sample就算WER
        wer_AD = compute_measures(truth=AD_result["text"], hypothesis=AD_result["pred_str"])
    else:
        wer_AD = {"wer": "No sample"}                                       # 沒有sample紀錄沒有

    if cluster_num != None:                                                 # 有分cluster就紀錄
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

    # 檢查檔案是否存在
    file_exists = os.path.isfile('./results/Overall_WER.csv')

    # 如果檔案存在，則追加數據時不需要header
    if file_exists:
        df.to_csv('./results/Overall_WER.csv', mode='a', header=False, index=False)
    else:
        # 如果檔案不存在，則創建新檔案並寫入header
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
            #model = Data2VecAudioForCTC_CPFL.from_pretrained(txt[0] + "_cluster" + str(cluster_id) + txt[1]+"/final/", config=config, args=args)
            model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
            model_lst.append(model)
    else:                                                                   # load from args.model_in_path
        #model = Data2VecAudioForCTC_CPFL.from_pretrained(args.model_in_path + "/final/", config=config, args=args)
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

def save_weights(folder_to_save, weights):
    encoder_state_dict, decoder_state_dict = weights
    #torch.save(encoder_state_dict, folder_to_save + "/encoder_weights.pth")
    os.makedirs(folder_to_save, exist_ok=True)
    torch.save(decoder_state_dict, folder_to_save + "/decoder_weights.pth")

############################################################################################
# 切client、切train / test相關
############################################################################################
#先找這個spk的INV or PAR，取後面20%的資料出來（剩下是training）
def split_train_test_spk(source_dataset, client_spk, identity, DEV):
    # identity: "INV" or "PAR"
    subsetA = source_dataset.filter(lambda example: example["path"].startswith(client_spk+"_"+identity))
                                                    # 先對這個spk的INV or PAR做處理
    #print(client_spk+"_"+identity)
    #print(subsetA["path"])                          # filter兩次看上去正常
    subsetA = subsetA.sort("path")                  # 按說話的先後次序排序
    LEN_subsetA = len(subsetA)                      # 這個spk的INV or PAR有多少sample
    if DEV:
        num_sample_train = max(1, int(LEN_subsetA*0.7))       # 最少一個sample，最多70%的sample當training set
        num_sample_trainDev = max(1, int(LEN_subsetA*0.8))       # 最少一個sample，最多80%的sample當training + dev set
        #print(num_sample, " / ", LEN_subsetA)           # 確認數量
        
        if num_sample_train == 0:                             # 若training set長度為0
            train_dataset = Dataset.from_dict({})       # 回傳空的dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample_train))
                                                        # 前面的當training set
        if num_sample_train == num_sample_trainDev:     # 若dev set長度為0
            test_dataset = Dataset.from_dict({})        # 回傳空的dataset
        else:        
            test_dataset = subsetA.select(range(num_sample_train, num_sample_trainDev))
                                                        # 後面的當dev set
    else:
        num_sample = max(1, int(LEN_subsetA*0.8))       # 最少一個sample，最多80%的sample當training set
        #print(num_sample, " / ", LEN_subsetA)           # 確認數量
        
        if num_sample == 0:                             # 若training set長度為0
            train_dataset = Dataset.from_dict({})       # 回傳空的dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample))
                                                        # 前面的當training set
        if num_sample == LEN_subsetA:                   # 若testing set長度為0
            test_dataset = Dataset.from_dict({})        # 回傳空的dataset
        else:        
            test_dataset = subsetA.select(range(num_sample, LEN_subsetA))
                                                        # 後面的當testing set

    """
    # 一個一個加入dataset的作法
    test_dataset = source_dataset.filter(lambda example: example["path"].startswith(client_spk+"_"+identity+"_"+str(LEN_subsetA-1)))
                                                    # 最後一個sample當作testing set的第一個sample
    for i in range(num_sample-1):                   # 繼續取剩下的samples
        sample = source_dataset.filter(lambda example: example["path"].startswith(client_spk+"_"+identity+"_"+str(LEN_subsetA-2-i)))
                                                    # 倒數第二個 到 倒數第num_sample個
        test_dataset = test_dataset.add_item(sample[0])
                                                    # 加入新sample
    """
    return train_dataset, test_dataset

from datasets import concatenate_datasets
def concatenate_ds(datasetA, datasetB):
    if len(datasetA) != 0 and len(datasetB) != 0:   # 皆不為0，直接合併
        concatenated_dataset = concatenate_datasets([datasetA, datasetB])
        return concatenated_dataset
    
    # 至少有一個為0
    if len(datasetA) != 0:                          # A不為0就回傳A
        return datasetA    
    return datasetB                                 # B不為0就回傳B。或兩個皆為0，回傳B做代表

# 回傳這個client的train / test set
def split_train_test_client(client_spks, source_dataset, DEV=False):    # default: no dev
    # 先做第一個spk，分別抓出INV與PAR的training(80%) and testing(20%) data
    client_spk = client_spks[0]
    train_dataset_INV, test_dataset_INV = split_train_test_spk(source_dataset, client_spk, "INV", DEV)
    train_dataset_PAR, test_dataset_PAR = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)

    # 把這個spk的INV與PAR合併
    train_dataset_client = concatenate_ds(train_dataset_INV, train_dataset_PAR)
    #print(train_dataset_INV["path"], train_dataset_PAR["path"])
    #print(train_dataset_client["path"])             # 內容的確是train_dataset_INV + train_dataset_PAR
    test_dataset_client = concatenate_ds(test_dataset_INV, test_dataset_PAR)

    #print(len(train_dataset_client), len(test_dataset_client))
    for i in range(len(client_spks)-1):             # 一個個spk處理
        # 從第二個spk開始，分別抓出INV與PAR的training(80%) and testing(20%) data
        client_spk = client_spks[i+1]
        train_dataset_INV, test_dataset_INV = split_train_test_spk(source_dataset, client_spk, "INV", DEV)
        train_dataset_PAR, test_dataset_PAR = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)

        # 把這個spk的INV與PAR合併
        train_dataset_spk = concatenate_ds(train_dataset_INV, train_dataset_PAR)
        test_dataset_spk = concatenate_ds(test_dataset_INV, test_dataset_PAR)
        #print(len(train_dataset_spk), len(test_dataset_spk))

        # 把這個spk的資料合併到client資料中
        train_dataset_client = concatenate_ds(train_dataset_client, train_dataset_spk)
        test_dataset_client = concatenate_ds(test_dataset_client, test_dataset_spk)    

    #print(len(train_dataset_client), len(test_dataset_client)) # 數量上看起來沒問題
    return train_dataset_client, test_dataset_client

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
# Mode 1: 和現在一樣，分完群後client train就全部拿來訓練
# Mode 2: 分完群後client train一部分切出來當client test，按句子切
# Mode 3: 分完群後client train一部分切出來當client test，按人切（暫略
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
    if cluster_id == None:                                                                  # 不分群
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
    else:                                                                                   # 分群，filter 1次到位
        print("Generating client training set for cluster ", str(cluster_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)) and example['cluster_id'] == cluster_id)
    
    # Mode 1: 分完群後client train就全部拿來訓練，沒有client test
    if args.eval_mode == 1:                             
        return client_train_dataset, None
    # Mode 2: 分完群後client train一部分切出來當client test，按句子切。即"這群人訓練的模型，測在這群人身上（不同句子）"
    elif args.eval_mode == 2: 
        train_dataset_client, test_dataset_client = split_train_test_client(client_spks, client_train_dataset)
        return train_dataset_client, test_dataset_client
    elif args.eval_mode == 3:                       # 70% training & 10% dev
        train_dataset_client, dev_dataset_client = split_train_test_client(client_spks, client_train_dataset, DEV=True)
        return train_dataset_client, dev_dataset_client
    # default: 分完群後client train就全部拿來訓練，沒有client test
    return client_train_dataset, None 

def train_split_unsupervised(args, dataset, client_id, cluster_id):
    # generate sub- training set for given user-ID
    if client_id == 0:                                                                      # 37 PAR w/ 37 AD
        client_spks = ['adrso045', 'adrso068', 'adrso247', 'adrso142', 'adrso059', 'adrso216', 'adrso246', 'adrso197', 'adrso046', 'adrso250', 
                        'adrso092', 'adrso234', 'adrso122', 'adrso187', 'adrso054', 'adrso244', 'adrso130', 'adrso189', 'adrso053', 'adrso125', 
                        'adrso071', 'adrso236', 'adrso116', 'adrso063', 'adrso220', 'adrso237', 'adrso198', 'adrso106', 'adrso109', 'adrso188', 
                        'adrso192', 'adrso077', 'adrso025', 'adrso202', 'adrso232', 'adrso090', 'adrso215']
    elif client_id == 1:                                                                    # 36 PAR w/ 27 AD
        client_spks = ['adrso228', 'adrso126', 'adrso263', 'adrso078', 'adrso249', 'adrso110', 'adrso224', 'adrso039', 'adrso075', 'adrso267', 
                        'adrso112', 'adrso047', 'adrso270', 'adrso245', 'adrso233', 'adrso024', 'adrso316', 'adrso190', 'adrso014', 'adrso123', 
                        'adrso144', 'adrso032', 'adrso289', 'adrso093', 'adrso206', 'adrso056', 'adrso027', 'adrso060', 'adrso151', 'adrso072', 
                        'adrso310', 'adrso223', 'adrso033', 'adrso260', 'adrso253', 'adrso128']
    elif client_id == 2:                                                                    # 32 PAR w/ 16 AD
        client_spks = ['adrso160', 'adrso028', 'adrso098', 'adrso218', 'adrso298', 'adrso222', 'adrso031', 'adrso200', 'adrso141', 'adrso248', 
                        'adrso265', 'adrso229', 'adrso177', 'adrso055', 'adrso015', 'adrso157', 'adrso278', 'adrso007', 'adrso016', 'adrso209', 
                        'adrso286', 'adrso154', 'adrso164', 'adrso259', 'adrso277', 'adrso134', 'adrso212', 'adrso172', 'adrso008', 'adrso089', 
                        'adrso074', 'adrso205']
    elif client_id == 3:                                                                    # 28 PAR w/ 7 AD
        client_spks = ['adrso291', 'adrso035', 'adrso169', 'adrso049', 'adrso183', 'adrso257', 'adrso070', 'adrso153', 'adrso211', 'adrso302', 
                        'adrso178', 'adrso180', 'adrso036', 'adrso268', 'adrso043', 'adrso138', 'adrso281', 'adrso005', 'adrso266', 'adrso309', 
                        'adrso148', 'adrso017', 'adrso186', 'adrso274', 'adrso002', 'adrso283', 'adrso299', 'adrso003']
    elif client_id == 4:                                                                    # 28 PAR w/ 28 HC
        client_spks = ['adrso280', 'adrso173', 'adrso156', 'adrso273', 'adrso162', 'adrso307', 'adrso276', 'adrso300', 'adrso292', 'adrso264', 
                        'adrso158', 'adrso285', 'adrso262', 'adrso010', 'adrso159', 'adrso308', 'adrso165', 'adrso182', 'adrso170', 'adrso161', 
                        'adrso315', 'adrso261', 'adrso167', 'adrso152', 'adrso296', 'adrso312', 'adrso012', 'adrso168']
    else:
        print("Train with whole dataset!!")
        return dataset
    
    print("Generating client training set for client ", str(client_id), "...")
    if cluster_id == None:                                                                  # 不分群
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
    else:                                                                                   # 分群，filter 1次到位
        print("Generating client training set for cluster ", str(cluster_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)) and example['cluster_id'] == cluster_id)

    # Mode 1: 分完群後client train就全部拿來訓練，沒有client test
    if args.eval_mode == 1:                             
        return client_train_dataset, None
    # Mode 2: 分完群後client train一部分切出來當client test，按句子切
    # 對unsupervised data來說，沒有ground truth無法做eval
    #elif args.eval_mode == 2: 
    #    train_dataset_client, test_dataset_client = split_train_test_client(client_spks, client_train_dataset)
    #    return train_dataset_client, test_dataset_client
    # default: 分完群後client train就全部拿來訓練，沒有client test
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

        if (args.training_type == 1) or (args.training_type == 3):                                  # (supervised) or (semi then supervised)
            save_path += "_Training" + "Address"
        elif (args.training_type == 2) or (args.training_type == 4):                                # (semi-supervised) or (supervised then semi)
            save_path += "_Training" + "AddressoWhisper"
        else:
            save_path += "_Training" + "AddressoWhisperandAddress"
        if args.FL_type == 3: # FML
            save_path += "_localModel" # eval local model
        args.model_in_path = save_path
        
        get_overall_wer(args, global_test_dataset)                                                  # get WER for global testing set
        # Mode 2: 分完群後client train一部分切出來當client test，按句子切。即"這群人訓練的模型，測在這群人身上（不同句子）"
        # Mode 3: 類似2，只是train在70% data上，測在10% dev與20% test上
        if args.eval_mode == 2 or args.eval_mode == 3:                                              # 測20% test
            # filter出client i的local_test_dataset，不分群（但dataset有cluster_id）
            # supervised才有人工的ground truth label
            origin_eval_mode = args.eval_mode
            args.eval_mode = 2                                                                      # 測20% test
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="test")                        # get WER for each client's testing set
            args.eval_mode = origin_eval_mode                                                       # back to original eval_mode
        
        if args.eval_mode == 3:                                                                     # 測10% dev
            # filter出client i的local_test_dataset，不分群（但dataset有cluster_id）
            # supervised才有人工的ground truth label
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
    # Mode 2: 分完群後client train一部分切出來當client test，按句子切。即"這群人訓練的模型，測在這群人身上（不同句子）"
    # Mode 3: 類似2，只是train在70% data上，測在10% dev與20% test上
    if args.eval_mode == 2 or args.eval_mode == 3:                                                  # 也測在local test上
        for i in idxs_users:                                                                        # for all clients that perform training
            # filter出client i的local_test_dataset，不分群（但dataset有cluster_id）
            # supervised才有人工的ground truth label
            origin_eval_mode = args.eval_mode
            args.eval_mode = 2
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="test")                        # get WER for each client's testing set
            args.eval_mode = origin_eval_mode                                                       # back to original eval_mode
    
    if args.eval_mode == 3:                                                                     # 測10% dev
        for i in idxs_users:  
            # filter出client i的local_test_dataset，不分群（但dataset有cluster_id）
            # supervised才有人工的ground truth label
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)

            if len(test_dataset_client) == 0:                                                       # no testing sample for client i 
                get_overall_wer(args, global_test_dataset)                                          # get WER for global testing set
            else:
                get_overall_wer(args, test_dataset_client, test_data="dev")                         # get WER for each client's testing set
        
