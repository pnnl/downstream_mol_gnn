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
import os.path as op
import numpy as np
import pandas as pd
import json
import argparse
import torch
from torch_geometric.data import DataLoader
from utils.infer import force_magnitude_error, force_angular_error
from utils.audit import load_static_model
from utils.water_dataset import PrepackedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Path to trained model to analyze.')
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset to analyze.')
parser.add_argument('--split', required=True, type=str, help='Path to split file to analyze.')
parser.add_argument('--savedir', default='./results/property_prediction/', type=str, help='Directory to save results.')
input_args = parser.parse_args()

if not op.isdir(input_args.savedir):
    os.mkdir(input_args.savedir)

split_name = input_args.split.split('/')[-1].replace('.npz','')
dataset_name = input_args.dataset.split('/')[-1].replace('.hdf5','')
model_name = input_args.model.split('/')[-1].replace('.pt','')

# get trained model
net = load_static_model(input_args.model, device='cpu')

# load dataset
dataset = PrepackedDataset(None, 
                           input_args.split, 
                           input_args.dataset.split('/')[-1].replace('_data.hdf5',''), 
                           directory='/'.join(input_args.dataset.split('/')[:-1]))


test_data=dataset.load_data('test')
print(f'{len(test_data)} items in test dataset')

loader = DataLoader(test_data, batch_size=256, shuffle=False)

fme_all = np.array(())
fae_all = np.array(())

df = pd.DataFrame()
for data in loader:
    # get predicted values
    e = net(data)
    f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
    
    # force errors 
    fme  = force_magnitude_error(data.f, f).numpy()
    fae  = force_angular_error(data.f, f).numpy()
    
    fme_all = np.concatenate((fme_all, fme))
    fme_all = np.concatenate((fae_all, fae))
    
    # get means of individual samples
    start = 0
    fme_individual = []
    fae_individual = []
    for size in data.size.numpy()*3:
        fme_individual.append(np.mean(np.abs(fme[start:start+size])))
        fae_individual.append(np.mean(np.abs(fae[start:start+size])))
        start += size
    
    tmp = pd.DataFrame({'cluster_size': data.size.numpy(), 
                        'e_actual': data.y.numpy(), 'e_pred': e.detach().numpy(),
                        'fme_mae':fme_individual, 'fae_mae':fae_individual})
    df = pd.concat([df, tmp])

# in direction of pred
df['e_error'] = df['e_pred']-df['e_actual']

# normalize by cluster size
df['e_error_by_water'] = df['e_error']/df['cluster_size']
df['e_actual_by_water'] = df['e_actual']/df['cluster_size']
df['e_pred_by_water'] = df['e_pred']/df['cluster_size']
    
df.to_csv(op.join(input_args.savedir,f"{model_name}-{dataset_name}.csv"), index=False)
print(f'saved to {op.join(input_args.savedir,f"{model_name}-{dataset_name}.csv")}')

