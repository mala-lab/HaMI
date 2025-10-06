import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init



class HaMI(nn.Module):
    def __init__(self, n_features, batch_size):
        super(HaMI, self).__init__()
        self.batch_size = batch_size
        self.feature_size = [n_features, 256]
        
        self.net = nn.Sequential(
            nn.Linear(n_features, self.feature_size[1]),
            nn.BatchNorm1d(self.feature_size[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_size[1], 1),
            nn.Sigmoid()
        )


    def forward(self, ninput, ainput, n_layer, device):
        n_len_list = [hidden.shape[0] for hidden in ninput]
        if ainput:
            a_len_list = [hidden.shape[0] for hidden in ainput]
            out = torch.cat([torch.cat(ninput), torch.cat(ainput)]) 
        else:
            a_len_list = []
            out = torch.cat(ninput) 

        out = out[:,n_layer,:].to(device).float()
        scores = self.net(out).flatten()
        
        return scores, n_len_list, a_len_list

    

def MIL_loss(scores, n_len_list, a_len_list, device, k_ratio):

    len_thresh = 8
    sparsity = torch.tensor(0.).to(device)
    smooth = torch.tensor(0.).to(device)
    loss = torch.tensor(0.).to(device)
    start_idx = 0
    nor_max = []
    for n_len in n_len_list:
        s = scores[start_idx:start_idx+n_len]
        start_idx += n_len

        k = int(n_len * k_ratio)+1
        s_topk, idx_topk = torch.topk(s, k)
        s_topk_mean = s_topk.mean()
        nor_max.append(s_topk_mean)
        if n_len > len_thresh:
            smooth+= torch.sum((s[:(n_len-1)]-s[1:])**2)*0.0008
        else:
            smooth+=0


    abn_max=[]
    for i, a_len in enumerate(a_len_list):
        s = scores[start_idx:start_idx+a_len]
        start_idx += a_len

        k = int(a_len * k_ratio)+1
        s_topk, idx_topk = torch.topk(s, k)
        s_topk_mean = s_topk.mean()
        abn_max.append(s_topk_mean)

        if a_len > len_thresh:
            smooth+= torch.sum((s[:(a_len-1)]-s[1:])**2)*0.0008

        else:
            smooth+=0

        loss += (1.-s_topk_mean**2 + nor_max[i]**2)


    loss = loss/len(n_len_list) + smooth/len(n_len_list)
    return loss