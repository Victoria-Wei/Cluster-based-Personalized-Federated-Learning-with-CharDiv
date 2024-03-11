#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch

from transformers.training_args import TrainingArguments
from transformers import Trainer
from typing import Dict
import numpy as np
import os
import pandas as pd
from models import DataCollatorCTCWithPadding, Data2VecAudioForCTC_CPFL
from datasets import concatenate_datasets
import copy
from transformers import Data2VecAudioConfig, Wav2Vec2Processor
from tensorboardX import SummaryWriter
from utils import reorder_col, add_cluster_id, load_model
import pickle
import shutil
from utils import train_split_supervised, train_split_unsupervised

LOG_DIR = './logs/' #log/'

CPFL_codeRoot = os.environ.get('CPFL_codeRoot')
CPFL_dataRoot = os.environ.get('CPFL_dataRoot')

from datasets import load_metric
wer_metric = load_metric("wer")
def create_compute_metrics(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metrics

class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            #dementia_labels = inputs.pop("dementia_labels") # pop 出來就會不見?
            
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
from collections import Counter
def map_to_result(batch, processor, model, idx):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        labels = torch.tensor(batch["labels"]).unsqueeze(0)  
        logits = model(input_values, labels=labels, EXTRACT=True).logits                        # includes ASR logits, hidden_states_mean, loss
                                                                                                # output_attentions=True,
        asr_lg = logits['ASR logits']
    
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]                                     # predicted transcript
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)                       # ground truth transcript
    
    hidden_states_mean = logits["hidden_states_mean"].tolist()                                  # [batch_size, hidden_size]
    
    # 計算每個"字母"出現的比例（次數/time-step），由大排到小
    flatten_arr = [item for sublist in pred_ids.numpy() for item in sublist]
    counter = Counter(flatten_arr)
    #print("len(flatten_arr): ", len(flatten_arr), ". len(pred_ids[0]): ", len(pred_ids.numpy()[0])) # 長度相等
    sorted_counter = counter.most_common()                                                      # 由大排到小

    vocab_ratio_rank = [0] * 32                                                                 # initialize
    i = 0
    for num, count in sorted_counter:                                                           # num: 字母的id，count: 出現次數
        vocab_ratio_rank[i] = count / len(flatten_arr)                                          # 轉換成"比例"
        i += 1                                                                                  # 換下一個

    # replace inf and nan with 999
    df = pd.DataFrame([logits["loss"].tolist()])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    loss = df.values.tolist()                                                                   # [batch_size, 1]

    entropy = [logits["entropy"]]                                                               # [batch_size, 1]
    """
    if "entropy_history" not in list(batch.keys()):                                             # create entropy list for the 1st time
        entropy = [logits["entropy"]]                                                           # [batch_size, 1]
    else:
        #print("shape of entropy_history: ", np.shape(np.array(batch["entropy_history"])))       # [batch_size, 1]
        entropy = list(np.concatenate((batch["entropy_history"], [logits["entropy"]]), axis=1))            # add new entropy                                        
        #print("shape of entropy_history: ", np.shape(np.array(entropy)), " after new ones added in") # [batch_size, 2(舊的+1)] 
    """    
    encoder_attention_1D = [logits["encoder_attention_1D"]]

    # for model w.o. FSM
    df = pd.DataFrame({'path': batch["path"],                                                   # to know which sample
                    #'array': str(batch["array"]),
                    'text': batch["text"],
                    'dementia_labels': batch["dementia_labels"],                                # used in detail_wer
                    #'input_values': str(batch["input_values"]),                                 # input of the model
                    #'labels': str(batch["labels"]),
                    #'ASR logits': str(logits["ASR logits"].tolist()),
                    #'dementia logits': str(logits["dementia logits"].tolist()),
                    #'hidden_states': str(logits["hidden_states"].tolist()),
                    #'pred_AD': batch["pred_AD"],                                                # AD prediction
                    'pred_str': batch["pred_str"]},
                    index=[idx])
    #print(np.array(entropy).shape)
    return df, hidden_states_mean, loss, entropy, [vocab_ratio_rank], encoder_attention_1D

def update_network_weight(args, source_path, target_weight, network):                           # update "network" in source_path with given weights
    # read source model                                                                         # return model   
    mask_time_prob = 0                                                                          # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
    model = Data2VecAudioForCTC_CPFL.from_pretrained(args.pretrain_name, config=config, args=args)
                                                                                                # use pre-trained model
    model.config.ctc_zero_infinity = True                                                       # to avoid inf values

    if network == "ASR":                                                                        # given weight from ASR
        data2vec_audio, lm_head = target_weight

        model.data2vec_audio.load_state_dict(data2vec_audio)                                    # replace ASR encoder's weight
        model.lm_head.load_state_dict(lm_head)                                                  # replace ASR decoder's weight

    return copy.deepcopy(model)

def get_model_weight(args, source_path, network):                                               # get "network" weights from model in source_path
    mask_time_prob = 0                                                                          # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
    #model = Data2VecAudioForCTC_CPFL.from_pretrained(source_path, config=config, args=args)     # load from source
    model = load_model(args, source_path, config)
    model.config.ctc_zero_infinity = True                                                       # to avoid inf values

    if network == "ASR":                                                                        # get ASR weights
        return_weights = [copy.deepcopy(model.data2vec_audio.state_dict()), copy.deepcopy(model.lm_head.state_dict())]
    
    return return_weights, copy.deepcopy(model)

class ASRLocalUpdate_CPFL(object):
    def __init__(self, args, dataset_supervised, dataset_unsupervised, global_test_dataset, client_id, cluster_id, model_in_path, model_out_path):
        self.args = args                                                                        # given configuration
        self.client_id = client_id                                                              # save client id
        self.cluster_id = cluster_id                                                            # save cluster id

        self.model_in_path = model_in_path                                                      # no info for client_id & global_round & cluster_id
        self.model_out_path = model_out_path   

        self.processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.device = 'cuda' if args.gpu else 'cpu'                                             # use gpu or cpu

        self.client_train_dataset_supervised=None
        self.client_train_dataset_unsupervised=None
        self.ALL_client_train_dataset_supervised=None
        self.ALL_client_train_dataset_unsupervised=None
        # if given dataset, get sub-dataset based on client_id & cluster_id
        if dataset_supervised is not None:
            self.client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, cluster_id)     # data of this client AND this cluster
            self.ALL_client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, None)     # data of this client       
            print("Da has ", len(self.client_train_dataset_supervised), " samples.")
        if dataset_unsupervised is not None:
            self.client_train_dataset_unsupervised, _ = train_split_unsupervised(args, dataset_unsupervised, client_id, cluster_id)    # data of this client AND this cluster
            self.ALL_client_train_dataset_unsupervised, _ = train_split_unsupervised(args, dataset_unsupervised, client_id, None)      # data of this client          
            print("Do has ", len(self.client_train_dataset_unsupervised), " samples.")
        self.client_test_dataset = global_test_dataset                                          # global testing set for evaluation
        if cluster_id != None:                                                                  # separate based on cluster_id
            self.client_test_dataset = self.test_split(global_test_dataset, client_id, cluster_id)  
    """                                                                                            # get subset of testing set (dataset of THIS cluster)
    def assign_cluster(self, sub_dataset):
        #sub_dataset = sub_dataset.reset_index(drop=True)

        # load ASR model
        mask_time_prob = 0                                                                      # change config to avoid code from stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = Data2VecAudioForCTC_CPFL.from_pretrained(self.model_in_path, config=config, args=self.args)
        processor = self.processor

        # load trained K-means model
        kmeans = pickle.load(open(self.args.Kmeans_model_path, 'rb'))

        # get emb.s ... 1 sample by 1 sample for sub_dataset
        _, hidden_states_mean = map_to_result(sub_dataset[0], processor, model, 0)
        for i in range(len(sub_dataset) - 1):
            _, hidden_states_mean_2 = map_to_result(sub_dataset[i+1], processor, model, i+1)
            hidden_states_mean.extend(hidden_states_mean_2)                                     # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
            print("\r"+ str(i), end="")
        
        # get cluster
        print("len of sub_dataset: ", len(sub_dataset)) # 419
        print("shape of embs: ", np.shape(np.array(hidden_states_mean))) # [419, 1024]
        cluster_id_lst = kmeans.predict(hidden_states_mean).tolist()                            # list of cluster_id
        print("len of predicted outcome: ", len(cluster_id_lst)) # 419
        # add to dataset
        #sub_dataset = sub_dataset.add_column("cluster_id", cluster_id_lst)
        #sub_dataset = sub_dataset.map(lambda x: add_cluster_id(x, cluster_id_lst[x.id]), with_indices=True)
        #sub_dataset = sub_dataset.map(lambda batch: add_cluster_id(batch, cluster_id_lst), batched=True)
        sub_dataset = sub_dataset.map(lambda example: add_cluster_id(example, cluster_id_lst.pop(0)))
        return sub_dataset
    
    def train_split_supervised(self, dataset, client_id, cluster_id):
        # generate sub- training set for given user-ID
        if client_id == "public":                                                               # get spk_id for public dataset, 54 PAR (50% of all training set)
            client_spks = ['S086', 'S021', 'S018', 'S156', 'S016', 'S077', 'S027', 'S116', 'S143', 'S082', 'S039', 'S150', 'S004', 'S126', 'S137', 
            'S097', 'S128', 'S059', 'S096', 'S081', 'S135', 'S094', 'S070', 'S049', 'S080', 'S040', 'S076', 'S093', 'S141', 'S034', 'S056', 'S090', 
            'S130', 'S092', 'S055', 'S019', 'S154', 'S017', 'S114', 'S100', 'S036', 'S029', 'S127', 'S073', 'S089', 'S051', 'S005', 'S151', 'S003', 
            'S033', 'S007', 'S084', 'S043', 'S009']                                             # 27 AD + 27 HC
        elif client_id == "public2":                                                            # get spk_id for public dataset, 54 PAR (50% of all training set) from clients
            client_spks = ['S058', 'S030', 'S064', 'S104', 'S048', 'S118', 'S122', 'S001', 'S087', 'S013', 'S025', 'S083', 'S067', 'S068', 'S111', 
            'S028', 'S015', 'S108', 'S095', 'S002', 'S072', 'S020', 'S148', 'S144', 'S110', 'S124', 'S129', 'S071', 'S136', 'S140', 'S145', 'S032', 
            'S101', 'S103', 'S139', 'S038', 'S153', 'S035', 'S011', 'S132', 'S006', 'S149', 'S041', 'S079', 'S107', 'S063', 'S061', 'S125', 'S062', 
            'S012', 'S138', 'S024', 'S052', 'S142']                                             # 27 AD + 27 HC
        elif client_id == 0:                                                                    # 10 PAR w/ 10 AD
            client_spks = ['S139', 'S125', 'S145', 'S149', 'S138', 'S144', 'S101', 'S136', 'S148', 'S108']
        elif client_id == 1:                                                                    # 12 PAR w/ 9 AD
            client_spks = ['S030', 'S124', 'S013', 'S111', 'S140', 'S095', 'S104', 'S006', 'S087', 'S153', 'S107', 'S142']
        elif client_id == 2:                                                                    # 10 PAR w/ 5 AD
            client_spks = ['S110', 'S028', 'S083', 'S038', 'S079', 'S067', 'S129', 'S052', 'S024', 'S132']
        elif client_id == 3:                                                                    # 12 PAR w/ 3 AD
            client_spks = ['S071', 'S012', 'S032', 'S103', 'S122', 'S118', 'S020', 'S015', 'S002', 'S041', 'S062', 'S072']
        elif client_id == 4:                                                                    # 10 PAR w/ 10 HC
            client_spks = ['S011', 'S025', 'S058', 'S001', 'S048', 'S064', 'S068', 'S063', 'S061', 'S035']                                    
        else:
            print("Train with whole dataset!!")
            return dataset

        print("Generating client training set for client ", str(client_id), "...")
        if cluster_id == None: # 不分群
            client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        else:
            print("Generating client training set for cluster ", str(cluster_id), "...")
            client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)) and example['cluster_id'] == cluster_id)
        
        return client_train_dataset
    
    def train_split_unsupervised(self, dataset, client_id, cluster_id):
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
        if cluster_id == None: # 不分群
            client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        else:
            print("Generating client training set for cluster ", str(cluster_id), "...")
            client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)) and example['cluster_id'] == cluster_id)
        
        return client_train_dataset
    """
    def test_split(self, dataset, client_id, cluster_id):
        # generate sub- testing set for given user-ID
        client_test_dataset = dataset
        """
        if client_id == "public":                                                               # get spk_id for public dataset, 24 PAR (50% of all testing set)
            client_spks = ['S197', 'S163', 'S193', 'S169', 'S196', 'S184', 'S168', 'S205', 'S185', 'S171', 'S204', 'S173', 'S190', 'S191', 'S203', 
                           'S180', 'S165', 'S199', 'S160', 'S175', 'S200', 'S166', 'S177', 'S167']                                # 12 AD + 12 HC

        elif client_id == 0:                                                                    # get spk_id for client 1, 12 PAR (25% of all testing set)
            client_spks = ['S198', 'S182', 'S194', 'S161', 'S195', 'S170', 'S187', 'S192', 'S178', 'S201', 'S181', 'S174']
                                                                                                # 6 AD + 6 HC
        elif client_id == 1:                                                                    # get spk_id for client 2, 12 PAR (25% of all testing set)  
            client_spks = ['S179', 'S188', 'S202', 'S162', 'S172', 'S183', 'S186', 'S207', 'S189', 'S164', 'S176', 'S206']
                                                                                                # 6 AD + 6 HC
        else:
            print("Test with whole dataset!!")
            return dataset
        
        print("Generating client testing set for client ", str(client_id), "...")
        client_test_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        """

        # generate sub- testing set for given cluster-ID
        if cluster_id != None:
            #print("Assigning clusters... ")
            #client_test_dataset = self.assign_cluster(client_test_dataset)
            print("Generating client testing set for cluster ", str(cluster_id), "...")
            client_test_dataset_k = client_test_dataset.filter(lambda example: example["cluster_id"]==cluster_id)
            return client_test_dataset_k
        
        return client_test_dataset
    
    def record_result(self, trainer, result_folder):                                            # save training loss, testing loss, and testing wer
        logger = SummaryWriter('./logs/' + result_folder.split("/")[-1])                        # use name of this model as folder's name

        for idx in range(len(trainer.state.log_history)):
            if "loss" in trainer.state.log_history[idx].keys():                                 # add in training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["loss"], trainer.state.log_history[idx]["epoch"]*100)

            elif "eval_loss" in trainer.state.log_history[idx].keys():                          # add in testing loss & WER, epoch*100 to obtain int
                logger.add_scalar('Loss/test', trainer.state.log_history[idx]["eval_loss"], trainer.state.log_history[idx]["epoch"]*100)
                logger.add_scalar('wer/test', trainer.state.log_history[idx]["eval_wer"], trainer.state.log_history[idx]["epoch"]*100)

            elif "train_loss" in trainer.state.log_history[idx].keys():                         # add in final training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["train_loss"], trainer.state.log_history[idx]["epoch"]*100)
        logger.close()

    def model_train(self, model, client_train_dataset, save_path, num_train_epochs):                              # train given model using given dataset, and save final result in save_path
                                                                                                # return model and its weights
        model.train()                                                                           # set to training mode

        training_args = TrainingArguments(
            output_dir=save_path,
            group_by_length=True,
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            evaluation_strategy="steps",
            num_train_epochs=num_train_epochs, #self.args.local_ep
            fp16=True,
            gradient_checkpointing=True, 
            save_steps=500, # 500
            eval_steps=self.args.eval_steps, # 500
            logging_steps=10, # 500
            learning_rate=self.args.learning_rate, # 1e-5 for ASR
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=1,
            log_level='debug',
            logging_strategy="steps",
            #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
            #fp16_full_eval=True,      # to save memory
            #max_grad_norm=0.5
        )
        #training_args.log_path='logs'
        compute_metrics_with_processor = create_compute_metrics(self.processor)
        trainer = CustomTrainer(
            model=model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=compute_metrics_with_processor,
            train_dataset=client_train_dataset,
            eval_dataset=self.client_test_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        if self.cluster_id != None:
            print(" | Client ", str(self.client_id), " cluster ", str(self.cluster_id)," ready to train! |")
        else:
            print(" | Client ", str(self.client_id), " ready to train! |")
        trainer.train()
        if self.args.STAGE == 1: # freeze all, train ASR decoder alone
            torch.save(copy.deepcopy(trainer.model.lm_head.state_dict()), save_path + "/decoder_weights.pth")
            #torch.save(copy.deepcopy(trainer.model.data2vec_audio.state_dict()), save_path + "/encoder_weights.pth")
            return_weights = [copy.deepcopy(trainer.model.data2vec_audio.state_dict()), copy.deepcopy(trainer.model.lm_head.state_dict())]
            result_model = trainer.model
        else:
            trainer.save_model(save_path + "/final")                                                # save final model
            return_weights = [copy.deepcopy(trainer.model.data2vec_audio.state_dict()), copy.deepcopy(trainer.model.lm_head.state_dict())]
            result_model = trainer.model
            # get "network" weights from model in source_path
            #return_weights, result_model = get_model_weight(args=self.args, source_path=save_path + "/final/", network="ASR")
        
        self.record_result(trainer, save_path)                                                  # save training loss, testing loss, and testing wer

        
        return return_weights, result_model
    
    def gen_addLogit_fn(self, model_global):
        def map_to_logit(batch):                                               # 一個batch只有一個sample
            with torch.no_grad():
                model = copy.deepcopy(model_global)
                # decode using corresponding model
                input_values = torch.tensor(batch["input_values"]).unsqueeze(0).to("cuda")
                model = model.to("cuda")
                logits = model(input_values).logits
                # save result
                batch["fix_logits"] = logits
            return batch
        return map_to_logit

    def update_weights(self, global_weights, global_round):
        # load training model (mutual model in FML)
        if self.args.FL_type != 3:
            if global_weights == None:                                                              # train from model from model_in_path
                mask_time_prob = 0                                                                  # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
                #model = Data2VecAudioForCTC_CPFL.from_pretrained(self.model_in_path, config=config, args=self.args)
                model = load_model(self.args, self.model_in_path[:-7], config)
                model.config.ctc_zero_infinity = True                                               # to avoid inf values
            else:                                                                                   # update train model using given weight
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                    # from model from model_in_path, update ASR's weight          
        elif self.args.FL_type == 3: # FML
            # initial local model
            mask_time_prob = 0                                                                  # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
            #self.args.fix_model = copy.deepcopy(model).to("cuda").eval()                        # model_mutual as reference for model_local
            self.args.FML_model = 0                                                             # 0 for local --> alpha for local

            path = self.model_in_path[:-7] + "_localModel/"
            if os.path.exists(path):                                                            # if local file exits
                model_local = load_model(self.args, path[:-1], config)                          # load local model
            else:
                model_local = load_model(self.args, self.model_in_path[:-7], config)            # or use the same as mutual
            model_local.config.ctc_zero_infinity = True                                         # to avoid inf values                                                                                                    

            # load mutual
            self.args.FML_model = 1                                                             # 1 for mutual --> beta for mutual
            #self.args.fix_model = copy.deepcopy(model_local).to("cuda").eval()                  # model_local as reference for model_mutual
            
            if global_weights == None:                                                          # train from model from model_in_path
                mask_time_prob = 0                                                              # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
                #model = Data2VecAudioForCTC_CPFL.from_pretrained(self.model_in_path, config=config, args=self.args)
                model_mutual = load_model(self.args, self.model_in_path[:-7], config)
                model_mutual.config.ctc_zero_infinity = True                                    # to avoid inf values
            else:                                                                               # update train model using given weight
                model_mutual = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                # from model from model_in_path, update ASR's weight                
        
        if self.client_id == "public":                                                          # train using public dataset
            save_path = self.model_out_path + "_global"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_supervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
            num_training_samples = len(self.client_train_dataset_supervised)

        elif self.args.training_type == 1:                                                      # supervised
            # save path for trained model (mutual model for FML)
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "Address"

            # CPFL use all training data from all cluster to train
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_supervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            
            if self.args.FL_type == 3: # FML: train model_local & model_mutal, only return model_mutual
                # train model_mutual
                print("trian model_mutual")
                #model_local.args.fix_model = None
                #self.args.fix_model = copy.deepcopy(model_local).to("cuda").eval()              # model_local as reference for model_mutual
                self.args.FML_model = 1                                                         # 1 for mutual
                dataset_mutual = dataset.map(self.gen_addLogit_fn(model_local))
                #print("dataset_mutual: ", dataset_mutual)
                return_weights, _ = self.model_train(model_mutual, dataset_mutual, save_path, num_train_epochs=self.args.local_ep)
                num_training_samples = len(self.client_train_dataset_supervised)
                #del model_mutual
                # remove previous model if exists
                if global_round > 0:
                    save_path_pre = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path_pre += "_cluster" + str(self.cluster_id)
                    save_path_pre += "_Training" + "Address"
                    shutil.rmtree(save_path_pre)

                # train model_local, and keep in local
                save_path += "_localModel"
                print("trian model_local")
                self.args.FML_model = 0                                                         # 0 for local
                #self.args.fix_model = copy.deepcopy(model).to("cuda").eval()                    # mutual as reference
                ##################
                # 這邊的model_mutual是訓練完成的！！
                ##################
                dataset_local = dataset.map(self.gen_addLogit_fn(model_mutual))
                self.model_train(model_local, dataset_local, save_path, num_train_epochs=self.args.local_ep)
                #num_training_samples = len(self.client_train_dataset_supervised)
                # remove previous model if exists
                if global_round > 0:
                    save_path_pre = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path_pre += "_cluster" + str(self.cluster_id)
                    save_path_pre += "_Training" + "Address_localModel"
                    shutil.rmtree(save_path_pre)
                #del model, model_local
            else:
                return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
                num_training_samples = len(self.client_train_dataset_supervised)
                # remove previous model if exists
                if global_round > 0:
                    save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path += "_cluster" + str(self.cluster_id)
                    save_path += "_Training" + "Address"
                    shutil.rmtree(save_path)
        else:
            print("other training_type, such as type ", self.args.training_type, " not implemented yet")
            aaa=ccc
        """
        # other training_type not implemented yet
        elif self.args.training_type == 2:                                                      # semi-supervised
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id) 
            save_path += "_Training" + "AddressoWhisper"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_unsupervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_unsupervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
            num_training_samples = len(self.client_train_dataset_unsupervised)
            # remove previous model if exists
            if global_round > 0:
                save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                if self.cluster_id != None:
                    save_path += "_cluster" + str(self.cluster_id)
                save_path += "_Training" + "AddressoWhisper"
                shutil.rmtree(save_path)
        elif self.args.training_type == 3:                                                      # semi then supervised
            # semi-supervised
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "AddressoWhisper"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_unsupervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_unsupervised
            _, model = self.model_train(model, dataset, save_path, num_train_epochs=int(self.args.local_ep / 2))
            shutil.rmtree(save_path)
            # supervised
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "Address"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_supervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=int(self.args.local_ep / 2))
            num_training_samples = len(self.client_train_dataset_supervised) + len(self.client_train_dataset_unsupervised)

            # remove previous model if exists
            if global_round > 0:
                save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                if self.cluster_id != None:
                    save_path += "_cluster" + str(self.cluster_id)
                save_path += "_Training" + "Address"
                shutil.rmtree(save_path)
        elif self.args.training_type == 4:                                                      # supervised then semi
            # supervised
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "Address"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_supervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            _, model = self.model_train(model, dataset, save_path, num_train_epochs=int(self.args.local_ep / 2))
            shutil.rmtree(save_path)
            # semi-supervised
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "AddressoWhisper"
            if self.args.CPFL:
                dataset = self.ALL_client_train_dataset_unsupervised                              # train with all client data
            else:
                dataset = self.client_train_dataset_unsupervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=int(self.args.local_ep / 2))
            num_training_samples = len(self.client_train_dataset_supervised) + len(self.client_train_dataset_unsupervised)
            # remove previous model if exists
            if global_round > 0:
                save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                if self.cluster_id != None:
                    save_path += "_cluster" + str(self.cluster_id)
                save_path += "_Training" + "AddressoWhisper"
                shutil.rmtree(save_path)
        else:                                                                                   # all together
            if self.args.training_type != 5:
                print("Training_type should be 1-5, which means supervised(1) / semi-supervised(2) / semi then supervised(3) / supervised then semi (4) / all together(5)")
                print("Current training_type: ", self.args.training_type, ". Will begin training w/ 'all together' instead.")
            # combine dataset
            supervised_cols = self.client_train_dataset_supervised.column_names
            unsupervised_cols = self.client_train_dataset_unsupervised.column_names

            if supervised_cols != unsupervised_cols:                                            # if col in different order, re-order
                train_dataset_unsupervised = reorder_col(self.client_train_dataset_supervised, self.client_train_dataset_unsupervised)
            else:
                train_dataset_unsupervised = self.client_train_dataset_unsupervised
            # 有可能有0 samples的狀況
            print("supervised dataset of client ", str(self.client_id) , " has ", len(self.client_train_dataset_supervised), "sample(s)")
            print("unsupervised dataset of client ", str(self.client_id) , " has ", len(train_dataset_unsupervised), "sample(s)")

            dataset_combine = concatenate_datasets([self.client_train_dataset_supervised, train_dataset_unsupervised])
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "AddressoWhisperandAddress"

            if self.args.CPFL:                          
                # combine dataset
                supervised_cols = self.ALL_client_train_dataset_supervised.column_names
                unsupervised_cols = self.ALL_client_train_dataset_unsupervised.column_names

                if supervised_cols != unsupervised_cols:                                            # if col in different order, re-order
                    train_dataset_unsupervised = reorder_col(self.ALL_client_train_dataset_supervised, self.ALL_client_train_dataset_unsupervised)
                else:
                    train_dataset_unsupervised = self.ALL_client_train_dataset_unsupervised

                dataset = concatenate_datasets([self.ALL_client_train_dataset_supervised, train_dataset_unsupervised])    # train with all client data            
            else:
                dataset = dataset_combine
                
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
            num_training_samples = len(dataset_combine)
            # remove previous model if exists
            if global_round > 0:
                save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                if self.cluster_id != None:
                    save_path += "_cluster" + str(self.cluster_id)
                save_path += "_Training" + "AddressoWhisperandAddress"
                shutil.rmtree(save_path)
        """
        return return_weights, num_training_samples                                             # return weight

    def extract_embs(self, TEST):                                                               # extract emb. using model in self.model_in_path
        # load model
        mask_time_prob = 0                                                                      # change config to avoid code from stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = load_model(self.args, self.model_in_path, config)
        processor = self.processor

        if TEST:
            # get emb.s... 1 sample by 1 sample for client test
            df, hidden_states_mean, loss, entropy, vocab_ratio_rank, _ = map_to_result(self.client_test_dataset[0], processor, model, 0)
            for i in range(len(self.client_test_dataset) - 1):
                df2, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, _ = map_to_result(self.client_test_dataset[i+1], processor, model, i+1)
                df = pd.concat([df, df2], ignore_index=True)
                hidden_states_mean.extend(hidden_states_mean_2)                                 # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
                loss.extend(loss2)                                                              # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
                entropy.extend(entropy2)
                vocab_ratio_rank.extend(vocab_ratio_rank2)
                #print("shape of extended list: ", np.shape(np.array(vocab_ratio_rank)))
                print("\r"+ str(i), end="")

            return df, hidden_states_mean, loss, entropy, vocab_ratio_rank
        else:
            hidden_states_mean_super = None
            hidden_states_mean_semi = None
            loss_super = None
            loss_semi = None
            entropy_super = None
            entropy_semi = None
            vocab_ratio_rank_super = None
            vocab_ratio_rank_semi = None
            encoder_attention_1D_super = None
            encoder_attention_1D_semi = None
            # get emb.s... 1 sample by 1 sample for client train
            #print("self.client_train_dataset_supervised: ", self.client_train_dataset_supervised)
            if (self.client_train_dataset_supervised != None) and (len(self.client_train_dataset_supervised) != 0):                                    # if given supervised dataset
                _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = map_to_result(self.client_train_dataset_supervised[0], processor, model, 0)
                for i in range(len(self.client_train_dataset_supervised) - 1):
                    _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2 = map_to_result(self.client_train_dataset_supervised[i+1], processor, model, i+1)
                    hidden_states_mean.extend(hidden_states_mean_2)                             # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
                    loss.extend(loss2)                                                          # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
                    entropy.extend(entropy2)                                                          # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
                    #print("shape of extended list: ", np.shape(np.array(entropy)))
                    vocab_ratio_rank.extend(vocab_ratio_rank2)
                    encoder_attention_1D.extend(encoder_attention_1D2)
                    print("\r"+ str(i), end="")
                hidden_states_mean_super = hidden_states_mean
                loss_super = loss
                entropy_super = entropy
                vocab_ratio_rank_super = vocab_ratio_rank
                encoder_attention_1D_super = encoder_attention_1D
            #print("self.client_train_dataset_unsupervised: ", self.client_train_dataset_unsupervised)
            if (self.client_train_dataset_unsupervised != None) and (len(self.client_train_dataset_unsupervised) != 0):                                  # if given unsupervised dataset
                _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = map_to_result(self.client_train_dataset_unsupervised[0], processor, model, 0) 
                for i in range(len(self.client_train_dataset_unsupervised) - 1):
                    _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2 = map_to_result(self.client_train_dataset_unsupervised[i+1], processor, model, i+1)
                    hidden_states_mean.extend(hidden_states_mean_2)                             # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
                    loss.extend(loss2)                                                          # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
                    entropy.extend(entropy2)
                    vocab_ratio_rank.extend(vocab_ratio_rank2)
                    encoder_attention_1D.extend(encoder_attention_1D2)
                    print("\r"+ str(i), end="")
                hidden_states_mean_semi = hidden_states_mean
                loss_semi = loss
                entropy_semi = entropy
                vocab_ratio_rank_semi = vocab_ratio_rank
                encoder_attention_1D_semi = encoder_attention_1D
            print("Training data Done")

            if (hidden_states_mean_super != None) and (hidden_states_mean_semi != None):
                hidden_states_mean_super.extend(hidden_states_mean_semi)                        # combine both dataset
                loss_super.extend(loss_semi)
                entropy_super.extend(entropy_semi)
                vocab_ratio_rank_super.extend(vocab_ratio_rank_semi)
                encoder_attention_1D_super.extend(encoder_attention_1D_semi)
                return hidden_states_mean_super, loss_super, entropy_super, vocab_ratio_rank_super, encoder_attention_1D_super
            elif hidden_states_mean_super != None:                                              # only supervised dataset exists
                return hidden_states_mean_super, loss_super, entropy_super, vocab_ratio_rank_super, encoder_attention_1D_super
            elif hidden_states_mean_semi != None:                                               # only unsupervised dataset exists
                return hidden_states_mean_semi, loss_semi, entropy_semi, vocab_ratio_rank_semi, encoder_attention_1D_semi
   

