from update_CPFL import ASRLocalUpdate_CPFL
import torch

def client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, 
                 test_dataset, idx, epoch, cluster_id, global_weights=None):          # train function for each client, train from model in model_in_path 
                                                                                      #                                                                   + "_global/final/"
                                                                                      # or model in last round
                                                                                      # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_cluster" + str(cluster_id)
                                                                                      #  + ("_Training" + "Address")
    # BUILD MODEL for every process
    torch.set_num_threads(1)
    if epoch == 0:                                                                    # start from global model
        model_in_path = model_in_path_root + "_global/final/"
    else:                                                                             # train from model in last round
        if args.training_type == 1:                                                   # supervised
            model_in_path = model_out_path + "_client" + str(idx) + "_round" + str(epoch-1)
            if cluster_id != None:
                model_in_path += "_cluster" + str(cluster_id) 
            model_in_path += "_Training" + "Address/final/"
    
    local_model = ASRLocalUpdate_CPFL(args=args, dataset_supervised=train_dataset_supervised, global_test_dataset=test_dataset, client_id=idx, 
                                      cluster_id=cluster_id, model_in_path=model_in_path, model_out_path=model_out_path)
                                                                                      # initial dataset of current client

    w, num_training_samples = local_model.update_weights(global_weights=global_weights, global_round=epoch) 
                                                                                      # from model_in_path model, update certain part using given weight
    
    torch.cuda.empty_cache()
    return w, num_training_samples

def centralized_training(args, model_in_path, model_out_path, train_dataset, test_dataset, epoch, client_id="public"):                    
                                                                                      # train function for global model, train from model in model_in_path
                                                                                      # final result in model_out_path + "_global/final"
    local_model = ASRLocalUpdate_CPFL(args=args, dataset_supervised=train_dataset, global_test_dataset=test_dataset, client_id=client_id, 
                        cluster_id=None, model_in_path=model_in_path+"final/", model_out_path=model_out_path)   
                                                                                      # initial public dataset
    local_model.update_weights(global_weights=None, global_round=epoch)               # from model_in_path to train


def client_getEmb(args, model_in_path, train_dataset_supervised, test_dataset, idx, cluster_id, TEST):                                                    
                                                                                      # function to get emb. for each client, from model in model_in_path +"/final/"
    torch.set_num_threads(1)
    local_model = ASRLocalUpdate_CPFL(args=args, dataset_supervised=train_dataset_supervised, global_test_dataset=test_dataset, client_id=idx, 
                                      cluster_id=cluster_id, model_in_path=model_in_path, model_out_path=None)
                                                                                      # initial dataset of current client & current cluster
    if TEST:
        df, hidden_states_mean, loss, entropy, vocab_ratio_rank = local_model.extract_embs(TEST)
        return df
    else:
        hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D = local_model.extract_embs(TEST)
        return hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D
