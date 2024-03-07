# Cluster-based Personalized Federated Learning (CPFL) with CharDiv

## Training by `bash run.sh`
1. Train Fine-tuned ASR $W_0^G$
   <details><summary>Show important arguments</summary>
   - FL_STAGE set to 1
   </details>

3. Perform K-means Clustering, resulting in K-means model ($KM$)
    <details><summary>Show important arguments</summary>
        - FL_STAGE: set to 3
        - check the clustering metric in sections [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L409) and [here](https://github.com/Victoria-Wei/Cluster-based-Personalized-Federated-Learning-with-CharDiv/blob/main/src/federated_main.py#L522)
    </details>
4. Perform CPFL
    <details><summary>Show important arguments</summary>
        - FL_STAGE set to 4
    </details>
## Inference by `bash run_extract.sh`
