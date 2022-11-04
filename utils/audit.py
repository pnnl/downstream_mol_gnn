"""
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor,
under Contract No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software
are reserved by DOE on behalf of the United States Government and the Contractor as provided in the Contract.
You are authorized to use this computer software for Governmental purposes but it is not to be released or
distributed to the public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  This notice including this sentence must appear on any
copies of this computer software.
"""

import numpy as np
from ase.geometry.analysis import Analysis
import pandas as pd
from operator import itemgetter
import os.path as op
from scipy.stats import ks_2samp, wasserstein_distance
from utils import graph, models
import torch
import ase
from ase.calculators.calculator import Calculator, all_changes
from torch.nn import functional as F
from torch_geometric.data import Data

def flatten(l):
    return [item for sublist in l for item in sublist]

##### STATIC AUDIT FUNCTIONS #####
def load_static_model(model_path, device='cpu'):
    # load state dict of trained model
    state=torch.load(model_path)
    
    # remove module. from statedict keys (artifact of parallel gpu training)
    state = {k.replace('module.',''):v for k,v in state.items()}

    # extract model params from model state dict
    num_gaussians = state['basis_expansion.offset'].shape[0]
    num_filters = state['interactions.0.mlp.0.weight'].shape[0]
    num_interactions = len([key for key in state.keys() if '.lin.bias' in key])

    # load model architecture
    net = models.SchNet(num_features = num_filters,
                        num_interactions = num_interactions,
                        num_gaussians = num_gaussians,
                        cutoff = 6.0)   

    # load trained weights into model
    net.load_state_dict(state)

    # set to eval mode
    net.eval().to(device)

    return net

##### CALCULATOR FUNCTIONS #####

def schnet_eg(atoms, net, device='cpu'):
    """
    Takes in ASE atoms and loaded net and predicts energy and gradients
    args: atoms (ASE atoms object), net (loaded trained Schnet model)
    return: predicted energy (eV), predicted gradients (eV/angstrom)
    """
    types = {'H': 0, 'O': 1}
    atom_types = [1, 8]
    
    #center = False
    #if center:
    #    pos = atoms.get_positions() - atoms.get_center_of_mass()
    #else:
    
    pos = atoms.get_positions()
    pos = torch.tensor(pos, dtype=torch.float) #coordinates
    size = int(pos.size(dim=0)/3)
    type_idx = [types.get(i) for i in atoms.get_chemical_symbols()]
    atomic_number = atoms.get_atomic_numbers()
    z = torch.tensor(atomic_number, dtype=torch.long)
    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(atom_types))
    data = Data(x=x, z=z, pos=pos, size=size, batch=torch.tensor(np.zeros(size*3), dtype=torch.int64), idx=1)

    data = data.to(device)
    data.pos.requires_grad = True
    e = net(data)
    f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0].cpu().data.numpy()
    e = e.cpu().data.numpy()

    return e.item()/23.06035, f/23.06035

class SchnetCalculator(Calculator):
    """ASE interface to trained model
    """
    implemented_properties = ['forces', 'energy']
    nolabel = True

    def __init__(self, best_model, atoms=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        state=torch.load(best_model)
    
        # remove module. from statedict keys (artifact of parallel gpu training)
        state = {k.replace('module.',''):v for k,v in state.items()}
        num_gaussians = state[f'basis_expansion.offset'].shape[0]
            
        num_filters = state[f'interactions.0.mlp.0.weight'].shape[0]
        num_interactions = len([key for key in state.keys() if '.lin.bias' in key])

        net = models.SchNet(num_features = num_filters,
                            num_interactions = num_interactions,
                            num_gaussians = num_gaussians,
                            cutoff = 6.0)

        net.load_state_dict(state)

        self.net = net
        self.atoms = atoms

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, gradients = schnet_eg(self.atoms, self.net)
        self.results['energy'] = energy
        self.results['forces'] = -gradients

##### Structure Calculations #####

def get_structure_metrics(cluster):
    """
    Params:
        cluster: ASE Atoms object
    Returns:
        OH covalent bonds (A)
        HOH bond angles (deg)
        OH_hbond_dists (A)
        HOH_hbond_angles (deg)
        OHO_hbond_angles (deg)   # https://pubs.acs.org/doi/10.1021/acs.biochem.8b00217 FIG 9
        OO_hbond_dist (A)        # https://pubs.acs.org/doi/10.1021/acs.biochem.8b00217 FIG 9
    """
    labels = ['OH_cov', 'HOH_cov', 'OH_hbond', 'HOH_hbond', 'OHO_hbond', 'OO_hbond']
    
    # get covalent bond distances and angles
    ana = Analysis(cluster)
    OH_cov_pairs = ana.get_bonds('O', 'H', unique=True)
    OH_covalent_bonds = ana.get_values(OH_cov_pairs)
    HOH_covalent_angles = ana.get_values(ana.get_angles('H', 'O', 'H', unique=True))
    #ana.clear_cache()
    
    OH_cov_pairs=OH_cov_pairs[0]
    
    # convert to H-bonding graph
    G = graph.create_graph(cluster)

    # filter out O-H covalent bonds
    OH_hbond_pairs = list(set(G.edges)-set(OH_cov_pairs))
    
    # stop if no H-bonds are found
    if len(OH_hbond_pairs) == 0:
        print('fully disconnected structure found')
        values = [OH_covalent_bonds[0], HOH_covalent_angles[0]]

        type_list=[]
        value_list=[]
        for i in range(len(values)):
            value_list.append(values[i])
            type_list.append([labels[i]]*len(values[i]))
        return pd.DataFrame({'type':flatten(type_list), 'value':flatten(value_list)})
        

    # compute H---O distances
    OH_hbond_dists = [cluster.get_distance(i,j) for i,j in OH_hbond_pairs]

    # get atomic numbers of OH hbond pairs
    OH_hbond_Z = [itemgetter(i,j)(cluster.get_atomic_numbers()) for i,j in OH_hbond_pairs]
    
    O_id = np.where(np.stack(OH_hbond_Z)==8)[1]
    O_nodes = [pair[O_id[i]] for i,pair in enumerate(OH_hbond_pairs)]

    # match those O node indices with their H in OH_conv_pairs to get ther Hs
    HOH_hbond_angles=[]
    for i,O in enumerate(O_nodes):
        bonded_H = list(set(flatten(itemgetter(np.where(np.stack(OH_cov_pairs)==O)[0])(np.stack(OH_cov_pairs)))))
        bonded_H.remove(O)
        angles = []
        for k in bonded_H: 
            try:
                # compute H---O-Hs angles for both H
                angles.append(cluster.get_angle(OH_hbond_pairs[i][0],OH_hbond_pairs[i][1],k))
            except:
                pass
        if len(angles)>0:
            # convert angles closest to 180deg
            angles = [a if a > 90 else 180-a for a in angles]
            HOH_hbond_angles+=angles

    H_id = np.where(np.stack(OH_hbond_Z)==1)[1]
    H_nodes = [pair[H_id[i]] for i,pair in enumerate(OH_hbond_pairs)]

    OHO_hbond_angles=[]
    OO_hbond_dist=[]
    for i,H in enumerate(H_nodes):
        bonded_O = list(set(flatten(itemgetter(np.where(np.stack(OH_cov_pairs)==H)[0])(np.stack(OH_cov_pairs)))))
        bonded_O.remove(H)
        try:
            # compute O-H---O angle 
            angle = cluster.get_angle(OH_hbond_pairs[i][0],OH_hbond_pairs[i][1],bonded_O[0]) 
            # convert angle closest to 180deg
            angle = angle if angle > 90 else 180-angle
            OHO_hbond_angles.append(angle)

            Hbonded_O = list(OH_hbond_pairs[i])
            Hbonded_O.remove(H)
            OO_hbond_dist.append(cluster.get_distance(Hbonded_O[0],bonded_O[0]))
        except:
            pass
    
    values = [OH_covalent_bonds[0], HOH_covalent_angles[0], OH_hbond_dists, HOH_hbond_angles, OHO_hbond_angles, OO_hbond_dist]
    
    type_list=[]
    value_list=[]
    for i in range(len(labels)):
        value_list.append(values[i])
        type_list.append([labels[i]]*len(values[i]))
        
    # return df
    return pd.DataFrame({'type':flatten(type_list), 'value':flatten(value_list)})
    
##### Distribution Comparison Methods #####

'''
Computing divergence for discrete variables
https://github.com/michaelnowotny/divergence
'''

def compute_probs(data, n=50): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = list(
                filter(
                    lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
                )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=50): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    # get bins ranging over both distributions 
    e, _ = compute_probs(np.concatenate((train_sample,test_sample), axis=0), n=n_bins)
    
    _, p = compute_probs(train_sample, n=e)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=50): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    
    # get bins ranging over both distributions 
    e, _ = compute_probs(np.concatenate((train_sample,test_sample), axis=0), n=n_bins)
    
    _, p = compute_probs(train_sample, n=e)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)

def compute_ks_statistic(train_sample, test_sample):
    return ks_2samp(test_sample, train_sample)[0]

def compute_wasserstein_distance(train_sample, test_sample):
    return wasserstein_distance(test_sample, train_sample)
