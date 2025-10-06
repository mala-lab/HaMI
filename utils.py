import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset

from sklearn import metrics
from collections import defaultdict


def load_embedding(args, mode='train'):
    datapath = f"{args.data_dir}/{mode}_{args.data_name}_{args.model_name}.pkl"
    acc_datapath = f"{args.data_dir}/acc_double_{mode}_{args.data_name}_{args.model_name}.pkl"
    se_datapath = f"{args.data_dir}/se_{mode}_{args.data_name}_{args.model_name}.pkl"
    loaded_data = pickle.load(open(datapath, 'rb'))
    acc_loaded_data = pickle.load(open(acc_datapath, 'rb'))
    se_loaded_data = pickle.load(open(se_datapath, 'rb'))
    assert list(loaded_data.keys()) == list(acc_loaded_data.keys()) == se_loaded_data['id'], "ID mismatch"

    # Processing train data
    se_prob_list = idx2prob(se_loaded_data['semantic_ids'], used_sample=args.used_sample) # turn class identity to probability - list of lists
    if mode == 'train':
        nor_hidden_list = []
        abn_hidden_list = []
    else:
        emb_hidden_list = defaultdict(list)
        label_hidden_list = defaultdict(list)
        random.seed(args.seed)
        val_indices = random.sample(range(len(loaded_data)), args.val_sample)
        print(f"val_indices: {len(val_indices)}")

    used_acc = 1 if mode == 'train' else args.used_sample
    for id, tid in enumerate(loaded_data):
        example = loaded_data[tid]
        high_t_answer = example['high_t_answer']
        for m in range(used_acc):
            acc = acc_loaded_data[tid][m]
            if acc == 2.0 and args.is_refined:
                continue

            hiddens = high_t_answer[m][-1][:, args.start_layer::args.interval_layer,:] if args.end_layer is None \
                else high_t_answer[m][-1][:,args.start_layer:args.end_layer:args.interval_layer,:]
            logits = high_t_answer[m][2]
            log_mean = np.mean(np.array(logits))
            se_prob = se_prob_list[id][m]
            # print(se_prob_list[id], se_prob)

            if args.mode == 'logits':
                logit_prob = [np.exp(logits[i]) for i in range(len(logits))]
                hiddens = hiddens*((args.a+args.b*torch.tensor(logit_prob)).reshape(-1,1,1))
            elif args.mode == 'log_mean':
                hiddens = hiddens*(args.a-args.b*log_mean)
            elif args.mode == 'se':
                hiddens = hiddens*(args.a+args.b*se_prob)
            elif args.mode == 'token_log_mean':
                token_log_mean = [np.mean(np.array(logits[:(i+1)])) for i in range(len(logits))]
                hiddens = hiddens*((args.a-args.b*torch.tensor(token_log_mean)).reshape(-1,1,1))
            
            if mode == 'train':
                if acc == 1.0 or acc == 2.0:
                    nor_hidden_list.append(hiddens)
                elif acc == 0.0:
                    abn_hidden_list.append(hiddens)

            else:
                if id in val_indices:
                    m='val'

                emb_hidden_list[m].append(hiddens)
                if acc == 2.0:
                    label_hidden_list[m].append(0.0)
                else:
                    label_hidden_list[m].append(1.0-acc)
    if mode == 'train':
        n_label = [0.0]*len(nor_hidden_list)
        a_label = [1.0]*len(abn_hidden_list)
        n_idx = list(range(len(nor_hidden_list)))
        a_idx = list(range(len(abn_hidden_list)))
        return nor_hidden_list, abn_hidden_list, n_label, a_label, n_idx, a_idx
    else:
        for m in range(used_acc):
            assert len(emb_hidden_list[m]) == len(label_hidden_list[m]), f"{m} Length mismatch"
        return emb_hidden_list, label_hidden_list
    


def idx2prob(semantic_ids, used_sample=6):
    """
    Converts semantic cluster results to probabilities.
    Args:
        semantic_ids (list): List of semantic_id lists.
        used_sample (int): Number of samples to use.
    Returns:
        list: List of probabilitity list for each question.
    """
    se_uncertainty = []
    for count in semantic_ids:
        if len(count) > used_sample:
            count = count[:used_sample]
        elif len(count) < used_sample:
            count = count
        values, nums = np.unique(count, return_counts=True)
        probabilities = nums / len(count)
        pro_dict = dict(zip(values, probabilities))
        prob_list = [pro_dict[i] for i in count]
        se_uncertainty.append(prob_list)

    return se_uncertainty



# ref: DeepOD: https://github.com/xuhongzuo/DeepOD/tree/main/deepod
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):

        self.x_data = torch.tensor(x_data, dtype=torch.long)  # features
        self.y_data = torch.tensor(y_data, dtype=torch.long)  # labels

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        # return a tuple of (x, y)
        return self.x_data[index], self.y_data[index]
    

def tabular_metrics(y_true, y_score):


    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return metrics.roc_auc_score(y_true, y_score), metrics.average_precision_score(y_true, y_score), f1