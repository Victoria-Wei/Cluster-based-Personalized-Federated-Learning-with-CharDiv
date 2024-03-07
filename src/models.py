#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F

import torch
from transformers import Data2VecAudioModel
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioPreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Data2VecAudioConfig, Wav2Vec2Processor

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

import numpy as np
import scipy
import copy

DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            <Tip warning={true}>
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [data2vec-audio-base](https://huggingface.co/facebook/data2vec-audio-base-960h), `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.
            </Tip>
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

_PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"

_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95

_CONFIG_FOR_DOC = "Data2VecAudioConfig"

class ReverseLayerF(torch.autograd.Function):
    def __init__(self):
        super(ReverseLayerF, self).__init__()
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
    
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), wf

class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target, AD_loss):
        input = input.to(torch.float)
        target = target.to(torch.int64)

        N, C = input.size()[:2]                                                         # [batch_size, 2]
        logpt = F.log_softmax(input, dim=1)
        pt = logpt.exp()                                                                # pred_prob: [batch_size, 2]
        #print("pt: ", pt)

        ## convert target (N, 1, *) into one hot vector (N, C, *)
        target = target.view(N, 1, -1)                                                  # (N, 1, *)
        last_size = target.size(-1)
        target_onehot = torch.zeros((N, C, last_size)).type_as(pt)                      # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)                                            # (N, C, *)

        true_positive = torch.sum(pt.view(N, C, last_size) * target_onehot, dim=2)      # (N, C): true label的預測"機率"
        total_target = torch.sum(target_onehot, dim=2)                                  # (N, C): true_prob

        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)           # (N, C): true label的預測"機率", false label為1
        # --> 目標把"各個class"對的抓出來
        total_predict = torch.sum(pt.view(N, C, last_size), dim=2)                      # (N, C): pred_prob for all labels
        precision = (true_positive + 1e-5) / (total_predict + 1e-5)                     # (N, C): true label為1，false label為機率微擾後的倒數
        #  --> 目標false label的機率越小越好
        f1 = 2 * recall * precision / (recall + precision)


        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
            #print("weight: ", self.weight)
            recall_ori = recall * self.weight * C                                       # (N, C): recall
            precision_ori = precision * self.weight * C                                 # (N, C): prec
            f1 = f1 * self.weight * C                                                           # (N, C): f1
            recall = (torch.ones((N, C)).type_as(recall) - recall) * self.weight * C            # (N, C): 1 - recall
            precision = (torch.ones((N, C)).type_as(precision) - precision) * self.weight * C   # (N, C): 1 - prec

        #print("recall: ", recall)
        recall_loss = torch.mean(recall)  # mean越小越好，recall越小越好，1 - true label的預測"機率"越小越好 --> true label的預測"機率"越大越好
        prec_loss = torch.mean(precision) # mean越小越好，precision越小越好，{1 - false label的機率微擾後的倒數} 越小越好 --> false label的機率越小越好
        f1_loss = 1 - torch.mean(f1)
        recall_ori_loss = 1 - torch.mean(recall_ori)
        precision_ori_loss = 1 - torch.mean(precision_ori)

        if AD_loss == "f1":
            return f1_loss
        elif AD_loss == "recall":
            return recall_loss
        elif AD_loss == "prec":
            return prec_loss
        elif AD_loss == "recall_ori":
            return recall_ori_loss
        elif AD_loss == "prec_ori":
            return precision_ori_loss
        
class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.alpha=torch.tensor(args.LAMBDA)
        #self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        #print("lm_thres = ", self.lm_thres)
        self.TOGGLE_RATIO=args.TOGGLE_RATIO
        self.GS_TAU=args.GS_TAU
        self.AD_loss=args.AD_loss
        if args.W_LOSS == None:                 # weight for HC and AD
            self.W_LOSS = [0.1, 0.9]                 # default weight for HC and AD
        else:
            self.W_LOSS = args.W_LOSS
        self.STAGE=args.STAGE

        # 加toggle network, lm_model
        #self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.arbitrator = nn.Linear(config.hidden_size, config.hidden_size*4)    # 2條保護AD資訊（one-hot後用其中一條），2條保護ASR資訊（one-hot後用其中一條）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        
        # 加dementia model
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        
        # define similarity loss: AM-Softmax, aka div loss
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if args.STAGE == 0:                                                     # freeze all, train ASR alone
            print("Current stage: 0")
            #self.freeze_data2vec_audio()
            #self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()            
        elif args.STAGE == 1:                                                  # freeze all, train AD classifier alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            #self.freeze_lm_fsm()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()
        elif args.STAGE == 2:                                                # freeze all, train toggle network alone
            print("Current stage: 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()   

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
    
    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
    
    def freeze_arbitrator(self):
        self.arbitrator.eval()
        for param in self.arbitrator.parameters():
            param.requires_grad = False       


    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 沒過FSM，用來單獨train AD classifier
        dementia_logits_unmask = self.dementia_head(hidden_states) # for stage 1 training

        # hidden_states: data2vec_audio embedding
        ###################
        # 製造mask
        ###################
        """
        m = nn.Sigmoid()
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
        """
        #m = nn.Sigmoid()
        #all_score = m(self.arbitrator(hidden_states))             # score range from 0~1
        all_score = self.arbitrator(hidden_states)
        """
        all_mask = torch.where(all_score >= self.lm_thres.to(all_score.device), torch.tensor(1.0).to(all_score.device), torch.tensor(0.0).to(all_score.device))                   # if condition, 1. else, 0
        all_mask = all_mask + 0 * self.arbitrator(hidden_states) # to have grad?  
        """
        # use Gunbel softmax
        #print(all_score)
        lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)     # first part for lm, size = [batch_size, time-step, hidden_state, 2]
        AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1) # second part for AD, size = [batch_size, time-step, hidden_state, 2]

        # toggle ratio
        if self.TOGGLE_RATIO != 0:                                                           # if toggle ratio is set
            # lm_score
            y0 = lm_score[:, :, :, 0]                                                   # target vector
            y1 = lm_score[:, :, :, 1]                                                   # another vector
            lm_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector
            # AD_score
            y0 = AD_score[:, :, :, 0]                                                   # target vector
            y1 = AD_score[:, :, :, 1]                                                   # another vector
            AD_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector      

        # go through GS to form mask
        #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        lm_mask = gumbel_softmax(lm_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]
        #AD_mask = torch.nn.functional.gumbel_softmax(AD_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        AD_mask = gumbel_softmax(AD_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states
        AD_masked = AD_mask*hidden_states

        ##############
        # head(clf)
        ##############
        """
        logits = self.lm_head(lm_masked)
        dementia_logits = self.dementia_head(lm_masked) # masked hidden state 過AD classifier
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)
        dementia_output_mean = torch.mean(dementia_logits_unmask,dim=1)
        """
        logits_unmask = self.lm_head(hidden_states)                                         # for fine-tune ASR
        logits = self.lm_head(lm_masked)                                                    # ASR loss
        dementia_logits = self.dementia_head(lm_masked)                                     # for AD GRL
        
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)   # for AD GRL
        dementia_output_mean_unmask = torch.mean(dementia_logits_unmask,dim=1)              # unmask

        logits_r = self.lm_head(AD_masked)                                                  # for ASR GRL
        dementia_logits = self.dementia_head(AD_masked)                                     # for AD classifier
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs_unmask = nn.functional.log_softmax(logits_unmask, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1) # logit轉prob
            log_probs_r = ReverseLayerF.apply(log_probs_r, self.alpha) # ASR-GRL
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_unmask = nn.functional.ctc_loss(
                    log_probs_unmask,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # ASR GRL
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                if self.AD_loss == "cel":
                    print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif self.AD_loss == "recall":                 
                    #print("loss: recall")
                    loss_fn = RecallLoss(weight=self.W_LOSS)                                                 # W_LOSS=[w_HC, w_AD]
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier: [batch_size, 2], [batch_size,]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask: [batch_size, 2], [batch_size,]
                    #print("dementia_output_mean_unmask: ", dementia_output_mean_unmask)
                    #print("dementia_labels: ", dementia_labels)
                    #print("dementia_loss: ", dementia_loss_unmask)
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse: [batch_size, 2], [batch_size,]

                elif self.AD_loss == "prec":                 
                    #print("loss: precision")
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                                      # emphasize on AD PAR
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse
                elif self.AD_loss == "f1":
                    #print("loss: f1")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse         
                elif self.AD_loss == "prec_ori":     
                    #print("loss: prec_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     
                elif self.AD_loss == "recall_ori":     
                    #print("loss: recall_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     

                # att loss
                #Att_loss = FSMatt_loss(lm_mask, AD_mask)                                                        # not used in this version
                # diversity loss: AM-Softmax
                #scores = torch.cat((hidden_states * lm_mask, hidden_states * AD_mask), dim=0)
                #am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #print("scores size: ", scores.size())
                #print("labels size: ", am_labels.size())
                #del hidden_states
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * AD_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
                
                #print("========================")
                #print(AD_mask, lm_mask)
                #print(loss, dementia_loss_rev, loss_r, dementia_loss, Att_loss, score_loss)
                if self.STAGE == 0:                                                     # fine-tune ASR
                    final_loss = loss_unmask
                elif self.STAGE == 1:                                                  # train AD classifier
                    #print("Current stage: 1")
                    final_loss = dementia_loss_unmask
                    #print("final loss: ", final_loss)
                elif self.STAGE == 2:                                                # train toggle network
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + score_loss #+ Att_loss #+ score_loss
                    #print(loss, dementia_loss_rev, loss_r, dementia_loss, l2_lambda * l2_norm)
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm

                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        return CausalLMOutput(
            loss=final_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class Data2VecAudioForCTC_eval(Data2VecAudioPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.alpha=torch.tensor(args.LAMBDA)
        #self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        #print("lm_thres = ", self.lm_thres)
        self.TOGGLE_RATIO=args.TOGGLE_RATIO
        self.GS_TAU=args.GS_TAU
        self.AD_loss=args.AD_loss
        if args.W_LOSS == None:                 # weight for HC and AD
            self.W_LOSS = [0.1, 0.9]                 # default weight for HC and AD
        else:
            self.W_LOSS = args.W_LOSS
        self.STAGE=args.STAGE

        # 加toggle network, lm_model
        #self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.arbitrator = nn.Linear(config.hidden_size, config.hidden_size*4)    # 2條保護AD資訊（one-hot後用其中一條），2條保護ASR資訊（one-hot後用其中一條）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        
        # 加dementia model
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        
        # define similarity loss: AM-Softmax, aka div loss
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if args.STAGE == 0:                                                     # freeze all, train ASR alone
            print("Current stage: 0")
            #self.freeze_data2vec_audio()
            #self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()            
        elif args.STAGE == 1:                                                  # freeze all, train AD classifier alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            #self.freeze_lm_fsm()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()
        elif args.STAGE == 2:                                                # freeze all, train toggle network alone
            print("Current stage: 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()   

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
    
    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
    
    def freeze_arbitrator(self):
        self.arbitrator.eval()
        for param in self.arbitrator.parameters():
            param.requires_grad = False       


    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 沒過FSM，用來單獨train AD classifier
        dementia_logits_unmask = self.dementia_head(hidden_states) # for stage 1 training

        # hidden_states: data2vec_audio embedding
        ###################
        # 製造mask
        ###################
        """
        m = nn.Sigmoid()
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
        """
        #m = nn.Sigmoid()
        #all_score = m(self.arbitrator(hidden_states))             # score range from 0~1
        all_score = self.arbitrator(hidden_states)
        """
        all_mask = torch.where(all_score >= self.lm_thres.to(all_score.device), torch.tensor(1.0).to(all_score.device), torch.tensor(0.0).to(all_score.device))                   # if condition, 1. else, 0
        all_mask = all_mask + 0 * self.arbitrator(hidden_states) # to have grad?  
        """
        # use Gunbel softmax
        #print(all_score)
        lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)     # first part for lm, size = [batch_size, time-step, hidden_state, 2]
        AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1) # second part for AD, size = [batch_size, time-step, hidden_state, 2]

        # toggle ratio
        if self.TOGGLE_RATIO != 0:                                                           # if toggle ratio is set
            # lm_score
            y0 = lm_score[:, :, :, 0]                                                   # target vector
            y1 = lm_score[:, :, :, 1]                                                   # another vector
            lm_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector
            # AD_score
            y0 = AD_score[:, :, :, 0]                                                   # target vector
            y1 = AD_score[:, :, :, 1]                                                   # another vector
            AD_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector      

        # go through GS to form mask
        #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        lm_mask = gumbel_softmax(lm_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]
        #AD_mask = torch.nn.functional.gumbel_softmax(AD_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        AD_mask = gumbel_softmax(AD_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states
        AD_masked = AD_mask*hidden_states

        ##############
        # head(clf)
        ##############
        """
        logits = self.lm_head(lm_masked)
        dementia_logits = self.dementia_head(lm_masked) # masked hidden state 過AD classifier
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)
        dementia_output_mean = torch.mean(dementia_logits_unmask,dim=1)
        """
        logits_unmask = self.lm_head(hidden_states)                                         # for fine-tune ASR
        logits = self.lm_head(lm_masked)                                                    # ASR loss
        dementia_logits = self.dementia_head(lm_masked)                                     # for AD GRL
        
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)   # for AD GRL
        dementia_output_mean_unmask = torch.mean(dementia_logits_unmask,dim=1)              # unmask

        logits_r = self.lm_head(AD_masked)                                                  # for ASR GRL
        dementia_logits = self.dementia_head(AD_masked)                                     # for AD classifier
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs_unmask = nn.functional.log_softmax(logits_unmask, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1) # logit轉prob
            log_probs_r = ReverseLayerF.apply(log_probs_r, self.alpha) # ASR-GRL
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_unmask = nn.functional.ctc_loss(
                    log_probs_unmask,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # ASR GRL
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                if self.AD_loss == "cel":
                    print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif self.AD_loss == "recall":                 
                    #print("loss: recall")
                    loss_fn = RecallLoss(weight=self.W_LOSS)                                                 # W_LOSS=[w_HC, w_AD]
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier: [batch_size, 2], [batch_size,]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask: [batch_size, 2], [batch_size,]
                    #print("dementia_output_mean_unmask: ", dementia_output_mean_unmask)
                    #print("dementia_labels: ", dementia_labels)
                    #print("dementia_loss: ", dementia_loss_unmask)
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse: [batch_size, 2], [batch_size,]

                elif self.AD_loss == "prec":                 
                    #print("loss: precision")
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                                      # emphasize on AD PAR
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse
                elif self.AD_loss == "f1":
                    #print("loss: f1")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse         
                elif self.AD_loss == "prec_ori":     
                    #print("loss: prec_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     
                elif self.AD_loss == "recall_ori":     
                    #print("loss: recall_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     
                # att loss
                #Att_loss = FSMatt_loss(lm_mask, AD_mask)                                                        # not used in this version
                # diversity loss: AM-Softmax
                #scores = torch.cat((hidden_states * lm_mask, hidden_states * AD_mask), dim=0)
                #am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #print("scores size: ", scores.size())
                #print("labels size: ", am_labels.size())
                #del hidden_states
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * AD_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
                
                #print("========================")
                #print(AD_mask, lm_mask)
                #print(loss, dementia_loss_rev, loss_r, dementia_loss, Att_loss, score_loss)
                if self.STAGE == 0:                                                     # fine-tune ASR
                    final_loss = loss_unmask
                elif self.STAGE == 1:                                                  # train AD classifier
                    #print("Current stage: 1")
                    final_loss = dementia_loss_unmask
                    #print("final loss: ", final_loss)
                elif self.STAGE == 2:                                                # train toggle network
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + score_loss #+ Att_loss #+ score_loss
                    #print(loss, dementia_loss_rev, loss_r, dementia_loss, l2_lambda * l2_norm)
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm

                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, "dementia_mask": AD_mask}
        
        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        AD_labels = [{"dementia_labels": feature["dementia_labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",                                   # to torch tensor
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["dementia_labels"] = torch.tensor([torch.tensor(d['dementia_labels']) for d in AD_labels]) # list of dict to list of tensor
        
        if "fix_logits" in features[0].keys():
            fix_logits = [{"fix_logits": feature["fix_logits"]} for feature in features]
            """
            for fix_logit in fix_logits:
                for logit in fix_logit["fix_logits"]:
                    for item in logit:
                        for d in item:
                            print(d)
                            test = torch.tensor(d)
                            aaa=ccc
            """
            #batch["fix_logits"] = torch.tensor([torch.tensor(d) for fix_logit in fix_logits for logit in fix_logit["fix_logits"] for item in logit for d in item]) # list of dict to list of tensor
            batch["fix_logits"] = torch.tensor([[[torch.tensor(d) for d in item] for item in logit] for fix_logit in fix_logits for logit in fix_logit["fix_logits"] ]) # list of dict to list of tensor

            #print("batch_size: ", batch["fix_logits"].size())
            #batch["fix_logits"] = fix_logits
        
        return batch
    
def get_entropy(inputs_prob):
    #print("inputs_prob size: ", inputs_prob.size())
    time_step, batch_size, _ = inputs_prob.size()                       # get input dim
    #print(inputs_prob)
    batch_entropy = []                                                  # record batch of entropy
    for i in range(batch_size):                                         # compute sample by sample
        entropy_sum = 0                                                 # set to 0
        for j in range(time_step):                                      # comput time-step by time-step
            #print(np.shape(np.array(inputs_prob[j][i])))
            prob = inputs_prob[j][i]
            #print(type(labels))
            if torch.is_tensor(prob):
                prob = prob.cpu().detach().numpy()
            entropy_sum += scipy.stats.entropy(prob, base=None)       # add to sum of entropy
            #print(i, j)
        #print(j)
        batch_entropy.append(entropy_sum / (j+1))                       # average over time
    #print("batch_entropy: ", batch_entropy)
    return batch_entropy

def prox_loss(model1: nn.Module, model2: nn.Module):
    prox_loss_ = 0
    for i, (w, w_t) in enumerate(zip(model1.parameters(), model2.parameters())):
        #if i in [0,1,2,3,4,5]: # 只算幾層?
        prox_loss_ += (w-w_t).norm(2)
    #print("prox_loss_: ", prox_loss_)
    #print("prox_loss_ type: ", type(prox_loss_))
    #aaa=ccc
    if torch.is_tensor(prox_loss_):
        loss = prox_loss_.item()
    else:
        loss = prox_loss_
    return loss

class Data2VecAudioForCTC_CBFL(Data2VecAudioPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.args = args
        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.STAGE=args.STAGE                                                    # current stage
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"

        # FedProx
        if args.FL_type == 2:                                                    # FedProx: save global model for loss
            print("Performing FedProx...")
            self.data2vec_audio_t = copy.deepcopy(self.data2vec_audio)
            self.dropout_t = copy.deepcopy(self.dropout)
            self.lm_head_t = copy.deepcopy(self.lm_head)
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if args.STAGE == 0:                                                      # freeze all, train ASR encoder & decoder
            print("Current stage: 0")
            #self.freeze_feature_projection()
            #self.freeze_data2vec_audio_encoder()
            #print("Freeze feature extractor, projection, and part of encoder")
            #self.freeze_data2vec_audio()
            #self.freeze_lm_head()      
        elif args.STAGE == 1:                                                    # freeze all, train ASR decoder alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()
            #self.freeze_lm_head()

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    def freeze_feature_projection(self):
        self.data2vec_audio.feature_projection.eval()
        for param in self.data2vec_audio.feature_projection.parameters():
            param.requires_grad = False

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_data2vec_audio_encoder(self):
        """
        data2vec_audio裡面有一個encoder，由num_hidden_layers層的Data2VecAudioEncoderLayer組成
        僅訓練最後一層
        """
        # 冻结所有层
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False
        #for layer in self.data2vec_audio.encoder.layers[:-1]:
        #    for param in layer.parameters():
        #        param.requires_grad = False

        # 最後24層（全部）的encoder可訓練
        # 最後12層可訓練，結果與最後一層可訓練一樣
        """
        for num_ly in range(24):
            for param in self.data2vec_audio.encoder.layers[-1*(num_ly+1)].parameters():
                param.requires_grad = True
        """
        for param in self.data2vec_audio.encoder.layers[-1].feed_forward.output_dense.parameters():
            param.requires_grad = True
        
    def LM_logit2loss(self, logits, reverse, labels, input_values, attention_mask, EXTRACT):
        ###################
        # 算loss for ASR
        # Input: logits, 要不要reverse, labels, input_values, attention_mask
        # Output: loss for ASR
        ###################
        #print("logits: ", logits.size())                                   # [batch_size, time-step, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        #print("log_probs: ", log_probs.size())                             # [time-step, batch_size, vocab_size]

        if reverse:
            log_probs = ReverseLayerF.apply(log_probs, self.alpha)          # ASR-GRL
        

        if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
            # log_probs: [time-step, batch_size, vocab_size]
            if EXTRACT:
                batch_entropy = get_entropy(np.exp(log_probs)) # turn log_probs into probs
                #batch_entropy = get_entropy(log_probs)
            #print("len batch_entropy", len(batch_entropy))
            else:
                batch_entropy = None

        return loss, batch_entropy
    
    def get_encoder_attention(self, encoder_attention):
        # outputs[-1] # [24, batch_size, 16, time-step, time-step]
        encoder_attention = encoder_attention[-1][0][-1][:][:] # for batch_size=1: [time-step, time-step] from last layer's last head
        if torch.is_tensor(encoder_attention):
            encoder_attention = encoder_attention.cpu().detach().numpy()
        encoder_attention = np.asarray(encoder_attention) 
        #print(encoder_attention.shape) # [time-step, time-step]
        time_step, _ = encoder_attention.shape

        if self.args.training_type == 1: # supervised
            time_steps_median = 130 # 129.5
        else: # 2 dataset combined
            time_steps_median = 149

        # 用成相同size
        if time_step < time_steps_median: # fill 0s
            new_shape = (int(time_steps_median), int(time_steps_median))
            new_arr = np.zeros(new_shape, dtype=encoder_attention.dtype) # array w/ all 0s
            new_arr[:time_step, :time_step] = encoder_attention         # first time_step*time_step is encoder_attention
        elif time_step > time_steps_median:
            new_arr = encoder_attention[:int(time_steps_median), :int(time_steps_median)]
                                                                        # clip to [time_steps_median, time_steps_median]
        else:
            new_arr = encoder_attention

        
        # 轉成1D
        axis_idx = 0 # 0: 前面壓掉, 1: 後面壓掉
        compress_type = "max" # 取Var, mean, min, max, median

        if compress_type == "var":
            encoder_attention_1D = np.var(new_arr, axis=axis_idx)
        elif compress_type == "mean":
            encoder_attention_1D = np.mean(new_arr, axis=axis_idx)
        elif compress_type == "min":
            encoder_attention_1D = np.min(new_arr, axis=axis_idx)
        elif compress_type == "max":
            encoder_attention_1D = np.max(new_arr, axis=axis_idx)
        elif compress_type == "median":
            encoder_attention_1D = np.median(new_arr, axis=axis_idx)
        elif compress_type == "flat":
            encoder_attention_1D = np.array([item for sublist in new_arr for item in sublist])
        #print("encoder_attention_1D.shape: ", encoder_attention_1D.shape)
        
        return encoder_attention_1D

    def get_fix_logits(self, input_values):
        model_device = input_values.device
        model = self.args.fix_model.to(model_device).eval()
        #fix_logits = self.args.fix_model(input_values.to(fix_model_device)).logits.to(model_device)         
        fix_logits = model(input_values).logits
        return fix_logits
    
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,                                                                                # 1 label
        dementia_labels=None,
        fix_logits=None,
        EXTRACT=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)                                                 # [batch_size, time-step, hidden_size]
        # 以上就是encoder的部分，後面就是把encoder出來的hidden_state拿來分配給後面的layer
        encoder_attention_1D = self.get_encoder_attention(outputs[-1])
        """
        print("outputs[-1]: ", outputs[-1]) # [24, batch_size, 16, time-step, time-step]
        包含所有self-attention layer的注意力權重 --> [num of self-attention layer (aka num_hidden_layers), batch_size, num_attention_heads, time-step, time-step]
        [time-step, time-step]：每個time-step對其他所有（包含自己）time-step的重要性
        print("len(outputs[-1]): ", len(outputs[-1])) # 24(num_hidden_layers)，每個hidden_layer通常包含一個self-attention layer
        print("len(outputs[-1][0]): ", len(outputs[-1][0])) # 1
        print("len(outputs[-1][0][0]): ", len(outputs[-1][0][0])) # 16(num_attention_heads), time-step, time-step
        print("len(hidden_states): ", len(hidden_states)) # 1
        print("len(hidden_states[0]): ", len(hidden_states[0])) # 48 (time-step)
        print("len(hidden_states[0][0]): ", len(hidden_states[0][0])) # 1024
        """
        logits = self.lm_head(hidden_states)                                                        # pass through decoder
        
        #print("logit get!!!!")
        # 算loss
        final_loss = None
        if (labels is not None) and (labels.numel() != 0):
            final_loss, batch_entropy = self.LM_logit2loss(logits, 0, labels, input_values, attention_mask, EXTRACT)
            if self.args.FL_type == 2:                      # FedProx
                final_loss = final_loss + self.args.mu/2 * prox_loss(self.data2vec_audio, self.data2vec_audio_t) \
                                        + self.args.mu/2 * prox_loss(self.dropout, self.dropout_t) \
                                        + self.args.mu/2 * prox_loss(self.lm_head, self.lm_head_t)
            elif self.args.FL_type == 3:
                #fix_model_device = self.args.fix_model.device
                #model_device = input_values.device
                #self.args.fix_model = self.args.fix_model.to(input_device)

                #fix_logits = self.args.fix_model(input_values.to(fix_model_device)).logits.to(model_device)                               # get other model's logit
                #print("fix_logits: ", np.shape.size())
                #print("input_values.size(): ", input_values.size())
                #print("logits.size(): ", logits.size())
                #fix_logits = self.get_fix_logits(input_values)
                KLdiv = nn.KLDivLoss(reduction='batchmean')
                log_prob = torch.log(F.softmax(logits, dim=2))
                fix_log_prob = torch.log(F.softmax(fix_logits, dim=2))
                kl_loss = KLdiv(log_prob, fix_log_prob) 
                #print("fix_logits: ", probabilities.size())
                #print("logits.size(): ", logits.size())     
                #print("fix_logits: ", torch.sum(probabilities[0][0]))
                #print("logits.size(): ", logits[0][0])                                               # compute KL divergence
                #print("kl_loss: ", kl_loss) # -335275.6875
                #print("final_loss: ", final_loss) # 48.8285
                #aaa=ccc
                #print("current FML_weight: ", self.args.FML_weight)
                if self.args.FML_model == 0: # local, use alpha
                    FML_weight = self.args.alpha
                elif self.args.FML_model == 1: # mutual, use beta
                    FML_weight = self.args.beta
                final_loss = FML_weight * final_loss + (1-FML_weight) * kl_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        
        if EXTRACT: # return vectors that we might need
            #logits_all = {'ASR logits': logits_list[0], 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
            #        'lm_mask': lm_mask, "dementia_mask": AD_mask}
            hidden_states_mean = torch.mean(hidden_states,dim=1) # [batch_size, time-step, hidden_size] --> [batch_size, hidden_size]
            logits_all = {'ASR logits': logits,  'hidden_states': hidden_states, 'hidden_states_mean': hidden_states_mean, "loss": final_loss, "entropy": batch_entropy,
                          'encoder_attention_1D': encoder_attention_1D}
        else:
            logits_all = logits

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
