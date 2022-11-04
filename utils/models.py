"""
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor,
under Contract No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software
are reserved by DOE on behalf of the United States Government and the Contractor as provided in the Contract.
You are authorized to use this computer software for Governmental purposes but it is not to be released or
distributed to the public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  This notice including this sentence must appear on any
copies of this computer software.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.models.schnet import GaussianSmearing, \
    InteractionBlock, ShiftedSoftplus
from torch_scatter.scatter import scatter_add
from torch_geometric.nn import knn_graph
import logging

def load_model(args, mode='train', device='cpu', frozen=False):
    net = load_pretrained_model(args, device=device, frozen=frozen)

    if mode=='eval':
        net.eval()
        print('model set to eval')

    return net


def load_pretrained_model(args, device='cpu', frozen=False):
    """
    Load single SchNet model
    """
    device = torch.device(device)
    
    # load state dict of trained model
    state=torch.load(args.start_model)
    
    # extract model params from model state dict
    num_gaussians = state['basis_expansion.offset'].shape[0]
    num_filters = state['interactions.0.mlp.0.weight'].shape[0]
    num_interactions = len([key for key in state.keys() if '.lin.bias' in key])
    
    # load model architecture
    net = SchNet(num_features = num_filters,
                 num_interactions = num_interactions,
                 num_gaussians = num_gaussians,
                 cutoff = args.cutoff)
    
    logging.info(f'model loaded from {args.start_model}')
    
    if args.load_state:
        # load trained weights into model
        net.load_state_dict(state)
        logging.info('model weights loaded')
    
    net.to(device)
    
    #register backward hook --> gradient clipping
    if not frozen:
        for p in net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))

    return net

class SchNet(nn.Module):
    def __init__(self,
                 num_features: int = 100,
                 num_interactions: int = 4,
                 num_gaussians: int = 25,
                 cutoff: float = 6.0,
                 max_num_neighbors: int = 28,
                 neighbor_method: str = 'knn',
                 batch_size: Optional[int] = None):
        """
        :param num_features (int): The number of hidden features used by both
            the atomic embedding and the convolutional filters (default: 100).
        :param num_interactions (int): The number of interaction blocks
            (default: 4).
        :param num_gaussians (int): The number of gaussians used in the radial
            basis expansion (default: 25).
        :param cutoff (float): Cutoff distance for interatomic interactions
            which must match the one used to build the radius graphs
            (default: 6.0).
        :param max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: 28)
        :param neighbor_method (str): Method to collect neighbors for each node.
            'knn' uses knn_graph; 'radius' uses radius_graph. 
            (default: 'knn')
        :param batch_size (int, optional): The number of molecules in the batch.
            This can be inferred from the batch input when not supplied.
        """
        super().__init__()
        self.num_features = num_features
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.neighbor_method = neighbor_method
        self.batch_size = batch_size

        self.atom_embedding = nn.Embedding(100,
                                           self.num_features,
                                           padding_idx=0)
        self.basis_expansion = GaussianSmearing(0.0, self.cutoff,
                                                self.num_gaussians)

        self.interactions = nn.ModuleList()

        for _ in range(self.num_interactions):
            block = InteractionBlock(self.num_features, self.num_gaussians,
                                     self.num_features, self.cutoff)
            self.interactions.append(block)

        self.lin1 = nn.Linear(self.num_features, self.num_features // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(self.num_features // 2, 1)

        self.reset_parameters()

    def hyperparameters(self):
        """
        hyperparameters for the SchNet model.
        :returns: Dictionary of hyperparamters.
        """
        return {
            "num_features": self.num_features,
            "num_interactions": self.num_interactions,
            "num_gaussians": self.num_gaussians,
            "cutoff": self.cutoff,
            "batch_size": self.batch_size
        }

    def extra_repr(self) -> str:
        """
        extra representation for the SchNet model.
        :returns: comma-separated string of the model hyperparameters.
        """
        s = []
        for key, value in self.hyperparameters().items():
            s.append(f"{key}={value}")

        return ", ".join(s)

    def reset_parameters(self):
        """
        Initialize learnable parameters used in training the SchNet model.
        """
        self.atom_embedding.reset_parameters()

        for interaction in self.interactions:
            interaction.reset_parameters()

        xavier_uniform_(self.lin1.weight)
        zeros_(self.lin1.bias)
        xavier_uniform_(self.lin2.weight)
        zeros_(self.lin2.bias)

    def forward(self, data):
        """
        Forward pass of the SchNet model
        :param z: Tensor containing the atomic numbers for each atom in the
            batch. Vector with size [num_atoms].
        :param edge_weight: Tensor containing the interatomic distances for each
            interacting pair of atoms in the batch. Vector with size [num_edges]
        :param edge_index: Tensor containing the indices defining the
            interacting pairs of atoms in the batch. Matrix with size
            [2, num_edges]
        :param batch: Tensor assigning each atom within a batch to a molecule.
            This is used to perform per-molecule aggregation to calculate the
            predicted energy. Vector with size [num_atoms]
        :param energy_target (optional): Tensor containing the energy target to
            use for evaluating the mean-squared-error loss when training.
        """
        pos = data.pos

        edge_index = knn_graph(
            data.pos,
            28,
            data.batch,
            loop=False,
        )

        row, col = edge_index

        edge_weight = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        edge_index = edge_index.view(2, -1).long()
        batch = data.batch.long()

        h = self.atom_embedding(data.z.long())
        edge_attr = self.basis_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)


        mask = (data.z == 0).view(-1, 1)
        h = h.masked_fill(mask.expand_as(h), 0.)

        batch = batch.view(-1)
        out = scatter_add(h, batch, dim=0, dim_size=self.batch_size).view(-1)

        return out




        
