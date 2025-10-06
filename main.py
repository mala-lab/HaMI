import os
import random
import argparse
import pickle
import json
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from hami import HaMI
from train import train
from validate import validate
from utils import *


FEATURES = {'llama3_8b': 4096,
            'llama3_70b': 8192, 
            'mistral_12b': 5120}


def init_seed(seed):
    random_seed = seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def main(args):
    # ================================== Train data =================================
    train_data_name = args.data_name
    nor_hidden_list, abn_hidden_list, n_label, a_label, n_idx, a_idx = load_embedding(args)

    # ================================== Test data =================================
    # aligning with SE settings, multiple responses are sampled for each question
    # about 400 questions are selected for evaluation
    test_dict = defaultdict(list)
    test_results = defaultdict(list)
    for data_name in args.test_data_list:
    # for data_name in [train_data_name]:
        args.data_name = data_name
        test_hidden_dict, test_label_dict = load_embedding(args, mode='validation')
        test_dict[args.data_name].append(test_hidden_dict)
        test_dict[args.data_name].append(test_label_dict)

    # ================================== Validation data ==============================
    val_hidden_list = test_dict[train_data_name][0]['val']
    val_label_list = test_dict[train_data_name][1]['val']

    # ================================== Experiments ================================
    init_seed(args.seed)
    args.data_name = train_data_name
    nor_loader = DataLoader(CustomDataset(n_idx,n_label), batch_size=args.batch_size, shuffle=True,drop_last=True)
    abn_loader = DataLoader(CustomDataset(a_idx,a_label), batch_size=args.batch_size, shuffle=True,drop_last=True)

    n_layer = nor_hidden_list[0].shape[1]
    val_results = []
    for idx_layer in range(n_layer):
        print(f"Layer {idx_layer} - Start training and evaluation")
        # =================================== 2.1 training ====================================
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HaMI(n_features=FEATURES[args.model_name], batch_size=args.batch_size)
        model.to(device)
        # model: final epoch model
        # model_best: best model for val data
        model, model_best, val_auc, val_best = train(args, nor_hidden_list, abn_hidden_list, nor_loader, abn_loader, model, \
                                                     device, idx_layer, val_hidden_list, val_label_list)
        val_results.append([val_best, val_auc])

        # =================================== 2.2 testing ===================================
        for data_name in args.test_data_list:
        # for data_name in [train_data_name]:
            test_hidden_dict = test_dict[data_name][0]
            test_label_dict = test_dict[data_name][1]
            test_score = []
            test_score_best = []
            for m in range(args.used_sample):
                test_hidden_list = test_hidden_dict[m]
                test_label_list = test_label_dict[m]
                test_auc = validate(model, test_hidden_list, test_label_list, idx_layer, device, args.k_ratio)
                test_auc_best = validate(model_best, test_hidden_list, test_label_list, idx_layer, device, args.k_ratio)
                test_score.append([test_auc])
                test_score_best.append([test_auc_best])
            test_score = np.mean(np.array(test_score), axis=0) # 1
            test_score_best = np.mean(np.array(test_score_best), axis=0)
            test_results[data_name].append(test_score)
            test_results[f"{data_name}_best"].append(test_score_best)
            print(f"{data_name} test: {test_score_best}, {test_score}")
        
        del model, model_best

    test_results['val'] = val_results
    # save score_dict
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(f"{args.save_dir}/{args.model_name}_{args.data_name}_{args.mode}_{args.seed}.pkl", 'wb') as f:
        pickle.dump(test_results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data loading
    parser.add_argument('--data_dir', type=str, default='your_data_directory', help='your data directory')
    parser.add_argument('--data_name', type=str, default='trivia_qa', help='trivia_qa, squad, nq, bioasq')
    parser.add_argument('--test_data_list', type=list, default=['trivia_qa', 'squad', 'nq', 'bioasq'])
    parser.add_argument('--model_name', type=str, default='llama3_8b', help='llama3_8b, llama3_70b, mistral_12b')
    parser.add_argument('--is_refined', type=bool, default=True)
    parser.add_argument('--val_sample', type=int, default=300)
    parser.add_argument('--mode', type=str, default='se', help='ori, logits, log_mean, se')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--end_layer', type=int, default=None)
    parser.add_argument('--interval_layer', type=int, default=1)
    parser.add_argument('--used_sample', type=int, default=5)
    
    # train & eval settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k_ratio', type=str, default=0.1)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    main(args)