# Cluster-based Personalized Federated Learning (CPFL) with CharDiv
This repo supports the paper "A Cluster-based Personalized Federated Learning Strategy for End-to-End ASR of Dementia Patients."
![CPFL_with_CharDiv_framework.png](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/framework.png)
The cluster-based personalized federated learning (**CPFL**) strategy groups samples with similar character diversity (**CharDiv**) into clusters using K-means model $KM$, and assigns clients to train these samples federally, creating a cluster-specific model for decoding others within the same cluster.

## Data preparation and preprocessing
We use [ADReSS challenge dataset](https://dementia.talkbank.org/ADReSS-2020/) as the training and testing sets. You have to join as a DementiaBank member to gain access to this dataset. Our input for ASR will be in utterance, segmented from the given session file. 

## Training by `bash run.sh`
The codes include training of **Fine-tuned ASR** $W_0^G$, **K-means model** $KM$, and $K$ cluster-specific models using CPFL.
1. Train Fine-tuned ASR $W_0^G$, used for extracting clustering metric
   * Important arguments
      - `FL_STAGE`: set to 1
      - `global_ep`: number of epoch for training Fine-tuned ASR $W_0^G$

2. Perform K-means Clustering, resulting in K-means model ($KM$)
   * Important arguments
      - `FL_STAGE`: set to 3
      - check the clustering metric in sections [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L157 "link") and [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L219 "link")

3. Perform CPFL, resulting in $K$ cluster-specific models
   * important arguments</summary>
      - `FL_STAGE`: set to 4

