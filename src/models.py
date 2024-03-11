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
            batch["fix_logits"] = torch.tensor([[[torch.tensor(d) for d in item] for item in logit] for fix_logit in fix_logits for logit in fix_logit["fix_logits"] ]) # list of dict to list of tensor
        
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
        #if i in [0,1,2,3,4,5]:
        prox_loss_ += (w-w_t).norm(2)

    if torch.is_tensor(prox_loss_):
        loss = prox_loss_.item()
    else:
        loss = prox_loss_
    return loss

class Data2VecAudioForCTC_CPFL(Data2VecAudioPreTrainedModel):
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
        elif args.STAGE == 1:                                                    # freeze all, train ASR decoder alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
        
    def LM_logit2loss(self, logits, labels, input_values, attention_mask, EXTRACT):
        ###################
        # compute loss for ASR
        # Input: logits, labels, input_values, attention_mask
        # Output: loss for ASR
        ###################
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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

        # fill to same size
        if time_step < time_steps_median: # fill 0s
            new_shape = (int(time_steps_median), int(time_steps_median))
            new_arr = np.zeros(new_shape, dtype=encoder_attention.dtype) # array w/ all 0s
            new_arr[:time_step, :time_step] = encoder_attention         # first time_step*time_step is encoder_attention
        elif time_step > time_steps_median:
            new_arr = encoder_attention[:int(time_steps_median), :int(time_steps_median)]
                                                                        # clip to [time_steps_median, time_steps_median]
        else:
            new_arr = encoder_attention

        
        # to 1D
        axis_idx = 0 # perform on dim 0
        compress_type = "max" # can be var, mean, min, max, median

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
        
        encoder_attention_1D = self.get_encoder_attention(outputs[-1])
        logits = self.lm_head(hidden_states)                                                        # pass through decoder
        
        # compute loss
        final_loss = None
        if (labels is not None) and (labels.numel() != 0):
            final_loss, batch_entropy = self.LM_logit2loss(logits, labels, input_values, attention_mask, EXTRACT)
            if self.args.FL_type == 2:                                                              # FedProx
                final_loss = final_loss + self.args.mu/2 * prox_loss(self.data2vec_audio, self.data2vec_audio_t) \
                                        + self.args.mu/2 * prox_loss(self.dropout, self.dropout_t) \
                                        + self.args.mu/2 * prox_loss(self.lm_head, self.lm_head_t)
            elif self.args.FL_type == 3:                                                            # FML
                KLdiv = nn.KLDivLoss(reduction='batchmean')
                log_prob = torch.log(F.softmax(logits, dim=2))
                fix_log_prob = torch.log(F.softmax(fix_logits, dim=2))
                kl_loss = KLdiv(log_prob, fix_log_prob) 

                if self.args.FML_model == 0:                                                        # local model, use alpha
                    FML_weight = self.args.alpha
                elif self.args.FML_model == 1:                                                      # mutual model, use beta
                    FML_weight = self.args.beta
                final_loss = FML_weight * final_loss + (1-FML_weight) * kl_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        
        if EXTRACT:                                                                                 # return vectors that we might need
            hidden_states_mean = torch.mean(hidden_states,dim=1)                                    # [batch_size, time-step, hidden_size] --> [batch_size, hidden_size]
            logits_all = {'ASR logits': logits,  'hidden_states': hidden_states, 'hidden_states_mean': hidden_states_mean, "loss": final_loss, "entropy": batch_entropy,
                          'encoder_attention_1D': encoder_attention_1D}
        else:
            logits_all = logits

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
