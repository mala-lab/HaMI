from tqdm import tqdm
import os
import copy
import pickle
import json
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from hami import MIL_loss
from validate import validate


def train(args, ndata, adata, nor_loader, abn_loader, model, device, idx_layer, val_hidden_list, val_label_list):
        
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6
    )
    loss_list = []
    auc_list=[]
    best_score = 0
    best_step = 0
    for step in tqdm(range(1, args.epoch + 1), desc="Training Epochs"):

        if (step - 1) % (len(nor_loader)) == 0:
            loadern_iter = iter(nor_loader)
        if (step - 1) % len(abn_loader) == 0:
            loadera_iter = iter(abn_loader) 

        with torch.set_grad_enabled(True):
            model.train()
            n_idx, nlabel = next(loadern_iter)
            a_idx, alabel = next(loadera_iter)
            ninput = [ndata[i] for i in n_idx]
            ainput = [adata[i] for i in a_idx]

            scores, n_len_list, a_len_list = model(ninput, ainput,n_layer=idx_layer, device=device)  # b*32  x 2048

            loss = MIL_loss(scores, n_len_list, a_len_list, device, args.k_ratio)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)


        val_auc = validate(model, val_hidden_list, val_label_list, idx_layer, device, args.k_ratio)
        auc_list.append(val_auc)
        if val_auc > best_score:
            best_score = val_auc
            best_step = step
            model_best = copy.deepcopy(model)
            # Save the model
            if not os.path.exists(f"{args.save_dir}/{args.model_name}"):
                os.makedirs(f"{args.save_dir}/{args.model_name}")
            torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}/{args.data_name}_{idx_layer}_{args.mode}_best.pth")

    # # save the final model
    torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}/{args.data_name}_{idx_layer}_{args.mode}_final.pth")

            
    return model, model_best, val_auc, best_score