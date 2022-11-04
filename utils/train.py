"""
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor,
under Contract No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software
are reserved by DOE on behalf of the United States Government and the Contractor as provided in the Contract.
You are authorized to use this computer software for Governmental purposes but it is not to be released or
distributed to the public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  This notice including this sentence must appear on any
copies of this computer software.
"""

import os
import logging
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from scipy.special import erfinv

import os 
import torch
import numpy as np
import pickle
from scipy.special import erfinv



def energy_forces_loss(data, p_energies, p_forces, energy_coeff):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    energies_loss = torch.mean(torch.square(data.y - p_energies))
    forces_loss = torch.mean(torch.square(data.f - p_forces))
    total_loss = (energy_coeff)*energies_loss + (1-energy_coeff)*forces_loss
    return total_loss, energies_loss, forces_loss




def train_energy_forces(model, loader, optimizer, energy_coeff, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch 
    """
    model.train()
    total_ef_loss = []

    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()
        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff)
        
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())

        ef_loss.backward()
        optimizer.step()
        
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    return ave_ef_loss



def get_error_distribution(err_list):
    """
    Compute the MAE and standard deviation of the errors in the examine set.
    """
    err_array = np.array(err_list)
    mae = np.average(np.abs(err_array))
    var = np.average(np.square(np.abs(err_array)-mae))
    return mae, np.sqrt(var)


def get_idx_to_add(net, examine_loader, optimizer,
                   mae, std, energy_coeff, 
                   split_file, al_step, device, min_nonmin,
                   max_to_add=0.15, error_tolerance=0.15,
                   savedir = './'):
    """
    Computes the normalized (by cluster size) errors for all entries in the examine set. It will add a max of
    max_to_add samples that are p < 0.15.
    """
    net.eval()
    all_errs = []
    for data in examine_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        energies_loss = torch.abs(data.y - e)
        f_red = torch.mean(torch.abs(data.f - f), dim=1)
        
        f_mean = torch.zeros_like(e)
        cluster_sizes = data['size'] #data.size
        for i in range(len(e)):            #loop over all clusters in batch
            energies_loss[i] /= cluster_sizes[i]
            f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])]))).clone().detach()
        
        total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
        total_err = total_err.tolist()
        all_errs += total_err
    
    with open(os.path.join(savedir, f'error_distribution_alstep{al_step}_{min_nonmin}.pkl'), 'wb') as f:
        pickle.dump(all_errs, f)    

    S = np.load(os.path.join(savedir, split_file))
    examine_idx = S["examine_idx"].tolist()
    
    cutoff = erfinv(1-error_tolerance) * std + mae
    n_samples_to_add = int(len(all_errs)*max_to_add)
    idx_highest_errors = np.argsort(np.array(all_errs))[-n_samples_to_add:]
    idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if all_errs[idx]>=cutoff]
    
    return idx_to_add


def get_pred_loss(model, loader, optimizer, energy_coeff, device, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []
    
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff)
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            energies_loss = torch.abs(data.y - e)
            f_red = torch.mean(torch.abs(data.f - f), dim=1)

            f_mean = torch.zeros_like(e)
            cluster_sizes = data['size'] #data.size
            for i in range(len(e)):            #loop over all clusters in batch
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err
    
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    
    if val == False:
        return ave_ef_loss
    
    else:
        mae, stdvae = get_error_distribution(all_errs) #MAE and STD from EXAMINE SET
        return ave_ef_loss, mae, stdvae
