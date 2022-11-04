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
import torch
import shutil
import logging
import json 
import csv
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils import data, models, train, eval, split, hooks

# import path arguments
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, required=True, help='Directory to save training results')
parser.add_argument('--args', type=str, required=True, help='Path to training arguments')
args = parser.parse_args()


# create directory to store training results
if not os.path.isdir(args.savedir):
    os.mkdir(args.savedir)
else:
    logging.warning(f'{args.savedir} will be overwritten')
    
if not os.path.isdir(os.path.join(args.savedir,'tensorboard')):
    os.mkdir(os.path.join(args.savedir,'tensorboard'))    
    
# set up tensorboard logger
writer = SummaryWriter(log_dir=os.path.join(args.savedir,'tensorboard'))

# copy args file to training folder
shutil.copy(args.args, os.path.join(args.savedir, 'args.json'))

# read in args
savedir = args.savedir
with open(args.args) as f:
    args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
args.savedir = savedir

# check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'model will be trained on {device}')

# get initial train, val, examine splits for dataset(s)
if args.create_splits:
    logging.info('creating new split(s)')
    # create split(s)
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            split.create_init_split(args.n_train_nonmin, 0.1, 0.1, 
                                    args.n_to_examine_nonmin, args.nonmin_db_size, 
                                    'nonmin', savedir=args.savedir)
            split.create_init_split(args.n_train_min, 0.005, 0.005, 
                                    args.n_to_examine_min, args.min_db_size, 
                                    'min', savedir=args.savedir)
    else:
        split.create_init_split(args.n_train_nonmin, 0.1, 0.1, 
                                args.n_to_examine_nonmin, args.nonmin_db_size, 
                                dataset, savedir=args.savedir)
else:
    # copy initial split(s) to savedir
    logging.info(f'starting from splits in {args.splitdir}')
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            shutil.copy(os.path.join(args.splitdir, f'split_00_{dataset}.npz'), 
                        os.path.join(args.savedir, f'split_00_{dataset}.npz'))
    else: 
        shutil.copy(os.path.join(args.splitdir, f'split_00_{args.datasets}.npz'), 
                    os.path.join(args.savedir, f'split_00_{args.datasets}.npz'))

# load datasets/dataloaders
train_loader, val_loader, examine_loaders = data.bulk_dataloader(args, split='00')

# load model
net = models.load_model(args, mode='train', device=device)
logging.info(f'model loaded from {args.start_model}')

#initialize optimizer and LR scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=args.start_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8, min_lr=0.000001)


# train
n_epochs = args.n_epochs_initial
total_epochs = 0

logging.info('beginning training...')
#while number_added > args.al_threshold:
for al_step in range(args.n_al_iters):
    logging.info(f'beginning active learning interation {al_step}')
    for e in range(n_epochs):
        # train model
        train_loss = train.train_energy_forces(net, train_loader, optimizer, args.energy_coeff, device)
        
        # get validation set loss
        if e == n_epochs-1 and args.mae_std_from_val == True:
            val_loss, mae, std = train.get_pred_loss(net, val_loader, optimizer, args.energy_coeff, device, val=True)
        else:
            val_loss = train.get_pred_loss(net, val_loader, optimizer, args.energy_coeff, device)
            
        scheduler.step(val_loss)
        
        # log training info
        writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)
        
        # on same plot
        writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)
        
        total_epochs+=1
                
    # Save current model
    if args.save_models:
        torch.save(net.state_dict(), os.path.join(args.savedir, f'finetune_ttm_alstep{al_step}.pt'))
        
    # on same plot
    writer.add_scalars('iteration_loss', {'train':train_loss,'val':val_loss}, al_step)
        
    # select new samples to add from examine set
    logging.info('choosing structures to add to training set')
    n_add = 0
    add_dict={}
    for i, dataset in enumerate(args.datasets):
        idx_to_add = train.get_idx_to_add(net, examine_loaders[i], optimizer,
                                          mae, std, args.energy_coeff,
                                          f'split_{str(al_step).zfill(2)}_{dataset}.npz', 
                                          al_step, device, dataset, 
                                          max_to_add = args.max_to_add,
                                          savedir = args.savedir)
        
        split.create_new_split(idx_to_add, args.n_to_examine[i], 
                               al_step, dataset, 
                               savedir=args.savedir)
        
        # store sample addition info
        add_dict[dataset]=len(idx_to_add)
        
    # log sample addition info
    writer.add_scalars('samples_added', add_dict, al_step)
    
    # Process new train and examine datasets
    train_loader, _, examine_loaders = data.bulk_dataloader(args, split=str(al_step+1).zfill(2))
    
    logging.info(f'iteration {al_step} complete')
    
    n_epochs = args.n_epochs_al
        
        
        
logging.info('beginning final training epochs')

# implement early stopping
early_stopping = hooks.EarlyStopping(patience=50, verbose=True, 
                                     path = os.path.join(args.savedir, 'finetune_ttm_final.pt'),
                                     trace_func=logging.info)

for i in range(args.n_epochs_end):
    train_loss = train.train_energy_forces(net, train_loader, optimizer, args.energy_coeff, device)
    val_loss = train.get_pred_loss(net, val_loader, optimizer, args.energy_coeff, device)

    scheduler.step(val_loss)
    
    # log training info
    writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)
    writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)
    
    total_epochs+=1
    
    # check for stopping point
    early_stopping(val_loss, net)
    if early_stopping.early_stop:
        break

writer.add_scalars('iteration_loss', {'train':train_loss,'val':val_loss}, args.n_al_iters+1)

# close tensorboard logger
writer.close()
