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
import numpy as np
import random

def create_new_split(idx_to_add, n_to_examine, iteration, suffix, savedir='./'):
    """
    Creates a new split files. Moving select indices from the reserve set to the training set and returning
    the remaining to the reserve set.
    """
    split_file = os.path.join(savedir, f"split_{str(iteration).zfill(2)}_{suffix}.npz")

    S = np.load(split_file)
    train_idx = S["train_idx"].tolist()
    reserve_idx = S["reserve_idx"].tolist()
    
    val_idx = S["val_idx"].tolist() #doesn't change
    test_idx = S["test_idx"].tolist() #doesn't change

    train_idx += idx_to_add
    reserve_idx = list(set(reserve_idx)-set(idx_to_add))
    
    random.shuffle(reserve_idx)
    examine_idx = reserve_idx[0:n_to_examine]

    new_iteration = str(iteration+1).zfill(2)  
    np.savez(os.path.join(savedir, f"split_{new_iteration}_{suffix}.npz"), 
             train_idx=train_idx, examine_idx=examine_idx,
             reserve_idx=reserve_idx, val_idx=val_idx, 
             test_idx=test_idx, add_idx=idx_to_add)
    
    
def create_init_split(n_train, val_frac, test_frac, n_to_examine, num_clusters, suffix, savedir='./'):
    """
    Function creates an initial split file with specified fractions of the dataset that should be
    assigned to the train, validation, test, and examine sets.
    """
    rand_idx = np.random.permutation(num_clusters)
    n_val = int(num_clusters*val_frac)
    n_test = int(num_clusters*test_frac)
    train_idx = rand_idx[0:n_train]
    val_idx = rand_idx[n_train:(n_train+n_val)]
    test_idx = rand_idx[(n_train+n_val):(n_train+n_val+n_test)]
    examine_idx = rand_idx[(n_train+n_val+n_test):(n_train+n_val+n_test+n_to_examine)]
    reserve_idx = rand_idx[(n_train+n_val+n_test+n_to_examine):]
    np.savez(os.path.join(savedir, f'split_00_{suffix}.npz'), 
             train_idx=train_idx, examine_idx=examine_idx,
             reserve_idx=reserve_idx, val_idx=val_idx, 
             test_idx=test_idx)


def new_init_split(args, dataset):
    """
    Single train/val/test/examine split for all datasets
    """
    cluster_list = np.random.permutation(args.n_total) 
    train_idx = cluster_list[0:args.n_train]
    val_idx = cluster_list[args.n_train:(args.n_train+args.n_val)]
    test_idx = cluster_list[(args.n_train+args.n_val):(args.n_train+args.n_val+args.n_test)]
    examine_idx = cluster_list[(args.n_train+args.n_val+args.n_test):(args.n_train+args.n_val+args.n_test+args.n_examine)]
    reserve_idx = cluster_list[(args.n_train+args.n_val+args.n_test+args.n_examine):]
    np.savez(os.path.join(args.savedir, f'split_00_{dataset}.npz'),
             train_idx=train_idx, 
             examine_idx=examine_idx,
             reserve_idx=reserve_idx,
             val_idx=val_idx,
             test_idx=test_idx)
