import torch
import numpy as np
from utils import tabular_metrics


def validate(model, val_hidden_list, val_label_list, idx_layer, device, k_ratio):
    model.eval()
    with torch.no_grad():
        y_score = []
        for val_hidden in val_hidden_list:
            scores, _, _ = model([val_hidden], None, n_layer=idx_layer, device=device)
            if k_ratio==1.0:
                k=val_hidden.shape[0]
            else:
                k = int(val_hidden.shape[0] * k_ratio)+1

            # k = 1
            top_k, idx_topk = torch.topk(scores, k)
            top_k_mean = top_k.mean()
            y_score.append(top_k_mean.cpu().item())

        
        auc, ap, f1 = tabular_metrics(np.array(val_label_list), np.array(y_score))

    return auc
    