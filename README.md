# Cluster-based Personalized Federated Learning (CPFL) with CharDiv

## Training by `bash run.sh`
1. Train Fine-tuned ASR $W_0^G$
   * Important arguments
      - `FL_STAGE`: set to 1

3. Perform K-means Clustering, resulting in K-means model ($KM$)
   * Important arguments
      - `FL_STAGE`: set to 3
      - check the clustering metric in sections [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L409 "link") and [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L522 "link")
      - check if CPFL (stage 4) was set to perform right after K-means clustering is done, in [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L751 "link")

4. Perform CPFL
   * important arguments</summary>
      - `FL_STAGE`: set to 4

