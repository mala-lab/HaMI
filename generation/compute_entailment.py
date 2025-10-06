import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35


def compute_entailment(args, dataset_split):
    entailment_model = EntailmentGPT35(None, False)

    filename = f"{args.data_dir}/qa_{dataset_split}_{args.data_name}_{args.model_name}.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Data: {filename}, {len(data)}")

    se_entropy = defaultdict(list) 

    for i, tid in tqdm(enumerate(data)):

        example = data[tid]
        question = example['question']
        answer = example['answer']
        high_t_answer = example['high_t_answer']
        res = [fr[0] for fr in high_t_answer]
        log_liks = [fr[1] for fr in high_t_answer]

        # compute semantic ids
        semantic_ids = get_semantic_ids(res, model=entailment_model, strict_entailment=True, example=example)
        
        se_entropy['id'].append(tid)
        se_entropy['semantic_ids'].append(semantic_ids)

        # compute entropy from frequencies of clustr assignments
        ce = cluster_assignment_entropy(semantic_ids)
        se_entropy['cluster_assignment_entropy'].append(ce)

        
        if dataset_split == 'validation':
            # compute semantic entropy
            # length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            se_entropy['semantic_entropy'].append(pe)

            # print(f"Semantic Entropy: {pe}")

        if (i + 1) % 100 == 0:
            filename = f"{args.data_dir}/se_{dataset_split}_{args.data_name}_{args.model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(se_entropy, f)
            print(f"Saved: {filename}")
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='your data directory')
    parser.add_argument('--data_name', type=str, default='squad', help='trivia_qa, squad, nq, bioasq')
    parser.add_argument('--model_name', type=str, default='llama3_70b')
    args = parser.parse_args()
    
    for data_split in ['train', 'validation']:
        # for args.data_name in ['trivia_qa', 'squad', 'nq', 'bioasq']:
        #     print(f"Computing entailment for {data_split} data: {args.data_name}")
        compute_entailment(args, data_split)
            



