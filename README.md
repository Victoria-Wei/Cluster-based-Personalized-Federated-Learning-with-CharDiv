# Cluster-based Personalized Federated Learning (CPFL) with CharDiv
This repo supports the paper "A Cluster-based Personalized Federated Learning Strategy for End-to-End ASR of Dementia Patients."
![CPFL_with_CharDiv_framework.png](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/framework.png)
The cluster-based personalized federated learning (**CPFL**) strategy groups samples with similar character diversity (**CharDiv**) into clusters using K-means model $KM$, and assigns clients to train these samples federally, creating a cluster-specific model for decoding others within the same cluster.

## Environment
Use `pip install -r requirements.txt` to install the same libraries

## Data preparation and preprocessing
> The examples provided below are merely illustrative; the information presented does not reflect actual data. Please follow the specified structure and modify the content accordingly to ensure accuracy.

We use [ADReSS challenge dataset](https://dementia.talkbank.org/ADReSS-2020/) as the training and testing sets. You have to join as a DementiaBank member to gain access to this dataset. Our input for ASR will be in utterance, segmented from the given session file using the time steps provided in the ground truth transcription. The information of each sample, including healthy control and dementia samples, as well as samples from investigators and participants, will be record in `train.csv` and `test.csv` with the following structure (an example is provided):
<pre><code>path, sentence
S987_INV_X_XXX_XXX.wav, this is an example
...
</code></pre>
where
* `path`: name of the file for the sample that ends with ".wav" and contains information for ID and the position (PAR for participant or INV for investigator) of the speaker, for example: `S987_INV_X_XXX_XXX.wav` is the utterance spoken by speaker 987's investigator
* `sentence`: ground truth transcription, for example:  `this is an example`
  
A dictionary mapping speaker ID to dementia label is also needed for analysis on separate groups of people. Generate the dictionary and assign the path to `path2_ADReSS_dict` [here](./src/utils.py#L81). For example:
<pre><code>ADReSS_dict = {
  "S987": 0,
  "S988": 0,
...
  "S900": 1
}
</code></pre>

## Training in 3 stages
The codes include the training of **three** components: the **Fine-tuned ASR** model $W_0^G$, the **K-means model** $KM$, and $K$ cluster-specific models using CPFL. **Each of these three stages of training needs to be conducted sequentially**, employing the following configurations, with:
<pre><code>bash run.sh
</code></pre>
1. Train Fine-tuned ASR $W_0^G$, used for extracting clustering metric
   * Important arguments
      - `FL_STAGE`: set to 1
      - `global_ep`: number of epoch for training Fine-tuned ASR $W_0^G$
      - `training_type`: only 1 (supervised) supported

2. Perform K-means Clustering, resulting in K-means model ($KM$)
   * Important arguments
      - `FL_STAGE`: set to 3
      - `training_type`: only 1 (supervised) supported
      - check the clustering metric in sections [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L158 "link") and [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L220 "link")

3. Perform CPFL, resulting in $K$ cluster-specific models
   * important arguments</summary>
      - `FL_STAGE`: set to 4
      - `training_type`: only 1 (supervised) supported
      - `N_Kmeans_update`: set using the same number as that of `epochs` to avoid re-clustering
      - `eval_mode`: set to 2 for 80% client training data and 20% client testing data, or set to 3 for 70% client training data and 10% client validation data

