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
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from utils.water_dataset import PrepackedDataset

def dataloader(datasets,
               mode,
               split = '00', 
               batch = 256,
               shuffle=True,
               splitdir = './data/splits/',
               datadir = './data/cached_dataset/'
               ):
    """
    dataset = str or list of str for each dataset label ['min', 'nonmin']
    mode = ['train','val','examine','test']
    """

    if isinstance(datasets, list):
        data = []
        for ds in datasets:
            dataset = PrepackedDataset(None, 
                                       op.join(splitdir,f'split_{split}_{ds}.npz'), 
                                       ds, 
                                       directory=datadir)
            data.append(dataset.load_data(mode))
            
        loader = DataLoader(ConcatDataset(data), batch_size=batch, shuffle=shuffle)
        
    else:
        dataset = PrepackedDataset(None, 
                                   op.join(splitdir,f'split_{split}_{datasets}.npz'), 
                                   datasets, 
                                   directory=datadir)
        data = dataset.load_data(mode)

        print(f'{datasets} data loaded from {datadir}')

        batch_size = batch if len(data) > batch else len(data)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader

def bulk_dataloader(args,
                    split = '00',
                    batch = 256,
                    shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """

    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]

    train_data = []
    val_data = []
    examine_data = []
    for ds in args.datasets:
        dataset = PrepackedDataset(None,
                                   op.join(args.savedir,f'split_{split}_{ds}.npz'),
                                   ds,
                                   directory=args.datadir)

        train_data.append(dataset.load_data('train'))
        val_data.append(dataset.load_data('val'))
        examine_data.append(dataset.load_data('examine'))

    train_loader = DataLoader(ConcatDataset(train_data), batch_size=batch, shuffle=shuffle)
    val_loader = DataLoader(ConcatDataset(val_data), batch_size=batch, shuffle=shuffle)
    examine_loaders = [DataLoader(ed, batch_size=batch, shuffle=shuffle) for ed in examine_data]

    return train_loader, val_loader, examine_loaders


def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f_actual=[]
    for size in data.size.numpy()*3:
        f_actual.append(np.abs(data.f[start:start+size].numpy()).max())   
        start += size
    return 
