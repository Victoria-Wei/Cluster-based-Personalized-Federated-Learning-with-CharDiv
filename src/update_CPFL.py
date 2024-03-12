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
from utils import train_split_supervised

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
    
    # compute freq. per character, and sort from largest to smallest
    flatten_arr = [item for sublist in pred_ids.numpy() for item in sublist]
    counter = Counter(flatten_arr)
    sorted_counter = counter.most_common()                                                      # sort from largest to smallest

    vocab_ratio_rank = [0] * 32                                                                 # initialize
    i = 0
    for num, count in sorted_counter:                                                           # num: char id，count: number of occurrence
        vocab_ratio_rank[i] = count / len(flatten_arr)                                          # convert to "ratio" (or freq.)
        i += 1                                                                                  # move to next char

    # replace inf and nan with 999
    df = pd.DataFrame([logits["loss"].tolist()])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    loss = df.values.tolist()                                                                   # [batch_size, 1]

    entropy = [logits["entropy"]]                                                               # [batch_size, 1]

    encoder_attention_1D = [logits["encoder_attention_1D"]]

    df = pd.DataFrame({'path': batch["path"],                                                   # to know which sample
                    'text': batch["text"],
                    'dementia_labels': batch["dementia_labels"],
                    'pred_str': batch["pred_str"]},
                    index=[idx])
    return df, hidden_states_mean, loss, entropy, [vocab_ratio_rank], encoder_attention_1D

def update_network_weight(args, source_path, target_weight, network):                           # update "network" in source_path with given weights
    # read source model                                                                         # return model   
    mask_time_prob = 0                                                                          # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
    model = Data2VecAudioForCTC_CPFL.from_pretrained(args.pretrain_name, config=config, args=args)
                                                                                                # use pre-trained model
    model.config.ctc_zero_infinity = True                                                       # to avoid inf values

    if network == "ASR":                                                                        # given weight for ASR
        data2vec_audio, lm_head = target_weight

        model.data2vec_audio.load_state_dict(data2vec_audio)                                    # replace ASR encoder's weight
        model.lm_head.load_state_dict(lm_head)                                                  # replace ASR decoder's weight

    return copy.deepcopy(model)

class ASRLocalUpdate_CPFL(object):
    def __init__(self, args, dataset_supervised, global_test_dataset, client_id, cluster_id, model_in_path, model_out_path):
        self.args = args                                                                        # given configuration
        self.client_id = client_id                                                              # save client id
        self.cluster_id = cluster_id                                                            # save cluster id

        self.model_in_path = model_in_path                                                      # no info for client_id & global_round & cluster_id
        self.model_out_path = model_out_path   

        self.processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.device = 'cuda' if args.gpu else 'cpu'                                             # use gpu or cpu

        self.client_train_dataset_supervised=None
        self.ALL_client_train_dataset_supervised=None
        # if given dataset, get sub-dataset based on client_id & cluster_id
        if dataset_supervised is not None:
            self.client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, cluster_id)         # data of this client AND this cluster
            self.ALL_client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, None)           # data of this client       
            print("Da has ", len(self.client_train_dataset_supervised), " samples.")
        self.client_test_dataset = global_test_dataset                                          # global testing set
    
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

    def model_train(self, model, client_train_dataset, save_path, num_train_epochs):            # train given model using given dataset, and save final result in save_path
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
        )

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
            return_weights = [copy.deepcopy(trainer.model.data2vec_audio.state_dict()), copy.deepcopy(trainer.model.lm_head.state_dict())]
            result_model = trainer.model
        else:
            trainer.save_model(save_path + "/final")                                            # save final model
            return_weights = [copy.deepcopy(trainer.model.data2vec_audio.state_dict()), copy.deepcopy(trainer.model.lm_head.state_dict())]
            result_model = trainer.model
        
        self.record_result(trainer, save_path)                                                  # save training loss, testing loss, and testing wer

        
        return return_weights, result_model
    
    def gen_addLogit_fn(self, model_global):
        def map_to_logit(batch):                                                                # only 1 sample per batch!!!!!!!!!
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
        # load training model
        if self.args.FL_type != 3:
            if global_weights == None:                                                              # train from model from model_in_path
                mask_time_prob = 0                                                                  # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
                model = load_model(self.args, self.model_in_path[:-7], config)
                model.config.ctc_zero_infinity = True                                               # to avoid inf values
            else:                                                                                   # update train model using given weight
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                    # from model from model_in_path, update ASR's weight          
        elif self.args.FL_type == 3:                                                                # FML
            # initial local model
            mask_time_prob = 0                                                                      # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
            self.args.FML_model = 0                                                                 # 0 for local --> alpha for local

            path = self.model_in_path[:-7] + "_localModel/"
            if os.path.exists(path):                                                                # if local file exits
                model_local = load_model(self.args, path[:-1], config)                              # load local model
            else:
                model_local = load_model(self.args, self.model_in_path[:-7], config)                # or use the same as mutual
            model_local.config.ctc_zero_infinity = True                                             # to avoid inf values                                                                                                    

            # load mutual
            self.args.FML_model = 1                                                                 # 1 for mutual --> beta for mutual
            
            if global_weights == None:                                                              # train from model from model_in_path
                mask_time_prob = 0                                                                  # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
                model_mutual = load_model(self.args, self.model_in_path[:-7], config)
                model_mutual.config.ctc_zero_infinity = True                                        # to avoid inf values
            else:                                                                                   # update train model using given weight
                model_mutual = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                    # from model from model_in_path, update ASR's weight                
        
        if self.client_id == "public":                                                              # train using public dataset
            save_path = self.model_out_path + "_global"
            if self.args.CBFL:
                dataset = self.ALL_client_train_dataset_supervised                                  # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
            num_training_samples = len(self.client_train_dataset_supervised)

        elif self.args.training_type == 1:                                                          # supervised
            # save path for trained model (mutual model for FML)
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "Address"

            # CBFL use all training data from all cluster to train
            if self.args.CBFL:
                dataset = self.ALL_client_train_dataset_supervised                                   # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            
            if self.args.FL_type == 3:                                                               # FML: train model_local & model_mutal, only return model_mutual
                # train model_mutual
                print("trian model_mutual")
                self.args.FML_model = 1                                                              # 1 for mutual
                dataset_mutual = dataset.map(self.gen_addLogit_fn(model_local))                      # local model as reference
                return_weights, _ = self.model_train(model_mutual, dataset_mutual, save_path, num_train_epochs=self.args.local_ep)
                num_training_samples = len(self.client_train_dataset_supervised)

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
                self.args.FML_model = 0                                                              # 0 for local

                dataset_local = dataset.map(self.gen_addLogit_fn(model_mutual))                      # mutual as reference
                self.model_train(model_local, dataset_local, save_path, num_train_epochs=self.args.local_ep)

                # remove previous model if exists
                if global_round > 0:
                    save_path_pre = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path_pre += "_cluster" + str(self.cluster_id)
                    save_path_pre += "_Training" + "Address_localModel"
                    shutil.rmtree(save_path_pre)
                #del model, model_local
            else:                                                                                    # train 1 model
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

        return return_weights, num_training_samples                                                  # return weight

    def extract_embs(self, TEST):                                                                    # extract emb. using model in self.model_in_path
        # load model
        mask_time_prob = 0                                                                           # change config to avoid code from stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = load_model(self.args, self.model_in_path, config)
        processor = self.processor

        if TEST:
            # get emb.s... 1 sample by 1 sample for client test
            df, hidden_states_mean, loss, entropy, vocab_ratio_rank, _ = map_to_result(self.client_test_dataset[0], processor, model, 0)
            for i in range(len(self.client_test_dataset) - 1):
                df2, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, _ = map_to_result(self.client_test_dataset[i+1], processor, model, i+1)
                df = pd.concat([df, df2], ignore_index=True)
                hidden_states_mean.extend(hidden_states_mean_2)                                      # [batch_size, hidden_size] + [batch_size, hidden_size] --> [2*batch_size, hidden_size]
                loss.extend(loss2)                                                                   # [batch_size, 1] + [batch_size, 1] --> [2*batch_size, 1]
                entropy.extend(entropy2)
                vocab_ratio_rank.extend(vocab_ratio_rank2)
                #print("shape of extended list: ", np.shape(np.array(vocab_ratio_rank)))
                print("\r"+ str(i), end="")

            return df, hidden_states_mean, loss, entropy, vocab_ratio_rank
        else:
            hidden_states_mean_super = None
            loss_super = None
            entropy_super = None
            vocab_ratio_rank_super = None
            encoder_attention_1D_super = None
            # get emb.s... 1 sample by 1 sample for client train
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
            print("Training data Done")

            return hidden_states_mean_super, loss_super, entropy_super, vocab_ratio_rank_super, encoder_attention_1D_super

   

