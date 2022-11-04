import torch
from torch.nn import functional as F
import numpy as np
import os.path as osp
from ase.db import connect
from typing import Optional, Callable
import gdown
import sys
import tempfile
import os
from copy import deepcopy as copy
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import DataLoader, InMemoryDataset, Data, extract_zip, download_url
import h5py
import logging

class PrepackedDataset(torch.utils.data.Dataset):
    def __init__(self, loader_list, split_file, dataset_type, shuffle=True, mode="train", directory="./data/cached_dataset/"):
        self.dataset = []
        self.shuffle = shuffle
        self.directory = directory
        self.mode = mode
        self.split_file = split_file
        self.dataset_type = dataset_type

        if not os.path.exists(directory):
            os.makedirs(directory)

        if loader_list is None:
            pass
        else:
            for i in range(len(loader_list)): 
                loader = loader_list[i]
                self.create_container(loader)
                logging.info("Finishing processing data...")
                bar = tqdm(enumerate(loader), total=len(loader))
                for index, data in bar:
                    self.x[index][:data.size*3, :] = copy(data.x).to(torch.uint8)
                    self.z[index][:data.size*3] = copy(data.z).to(torch.uint8)
                    self.pos[index][:data.size*3] = copy(data.pos).to(torch.float32)
                    self.y[index] = copy(data.y).to(torch.float32)
                    self.f[index][:data.size*3] = copy(data.f).to(torch.float32)
                    self.size[index] = copy(data.size).to(torch.uint8)
                self.save_data()

    def create_container(self, loader):
        tmp = next(iter(loader))
        n_elements = len(loader)
        batch, f, idx, name, pos, ptr, size, x, y, z = tmp
        force_coords = torch.zeros((90, 3))
        batch_size, x_size, z_size, pos_size, y_size, f_size, size_size = list(torch.zeros((90)).size()), list(torch.zeros((90,2)).size()), list(torch.zeros((90)).size()), list(force_coords.size()), list(y[1].size()), list(force_coords.size()), list(size[1].size())

        x_size.insert(0, n_elements)
        z_size.insert(0, n_elements)
        pos_size.insert(0, n_elements)
        y_size.insert(0, n_elements)
        f_size.insert(0, n_elements)
        size_size.insert(0, n_elements)

        self.x = np.zeros(x_size, dtype=np.uint8)
        self.z = np.zeros(z_size, dtype=np.uint8)
        self.pos = np.zeros(pos_size)
        self.y = np.zeros(y_size)
        self.f = np.zeros(f_size)
        self.size = np.zeros(size_size, dtype=np.uint8)

    def save_data(self):
        logging.info("Saving cached data in disk...")
        dataset = h5py.File(os.path.join(self.directory,f"{self.dataset_type}_data.hdf5"), "w")
        dataset.create_dataset("z", dtype=np.uint8, data=self.z)
        dataset.create_dataset("x", dtype=np.uint8, data=self.x)
        dataset.create_dataset("pos", dtype=np.float32, data=self.pos)
        dataset.create_dataset("y", dtype=np.float32, data=self.y)
        dataset.create_dataset("f", dtype=np.float32, data=self.f)
        dataset.create_dataset("size", dtype=np.uint8, data=self.size)
        dataset.close()

    def load_data(self, idx_type):
        logging.info("Loading cached data from disk...")
        dataset = h5py.File(os.path.join(self.directory, f"{self.dataset_type}_data.hdf5"), "r")

        S = np.load(self.split_file)
        self.mode_idx = S[f'{idx_type}_idx']

        data_list = []
        for i in range(len(self.mode_idx)):
            index = self.mode_idx[i]
            cluster_size = dataset["size"][index][0]
            
            z = torch.from_numpy(dataset["z"][index][:cluster_size*3])
            x = torch.from_numpy(dataset["x"][index][:cluster_size*3])
            pos = torch.from_numpy(dataset["pos"][index][:cluster_size*3])
            y = torch.from_numpy(dataset["y"][index])
            f = torch.from_numpy(dataset["f"][index][:cluster_size*3])
            size = torch.from_numpy(dataset["size"][index])
            data = Data(x=x, z=z, pos=pos, y=y, f=f, size=size)
            data_list.append(data)
        
        return data_list

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return self.x[index], self.z[index], self.pos[index], self.y[index], self.f[index], self.size[index]

    def get_dataloader(self, options):
        return DataLoader(data, batch_size=1, shuffle=self.shuffle)

    
class WaterDataSet(InMemoryDataset):
    raw_url = None

    def __init__(self,
                 sample,
                 set_type,
                 split_file,
                 iteration,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        """
        Args:
            sample: Dataset name (no extension)
            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        self.atom_types = [1, 8]
        
        self.set_type = set_type
        self.iteration = iteration
        self.split_file = split_file
        self.root = root
        self.sample = sample
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # NB: database file
        return f'{self.sample}.db'

    @property
    def processed_file_names(self):
        return f'{self.sample}_{self.set_type}{self.iteration}.pt'

    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """
        """
        raw_file_path = osp.join(self.raw_dir, self.raw_file_names)
        if osp.exists(raw_file_path):
            print(f'Using existing file {self.raw_file_names}',
                  file=sys.stderr)
            return
        else:
            with tempfile.TemporaryFile() as fp:
                extract_zip(gdown.download(self.raw_url, fp), self.raw_dir)
        """
        print('no download')

    def process(self):
        """
        Processes the raw data and saves it as a Torch Geometric data set
        The steps does all pre-processing required to put the data extracted from the database into graph 'format'. Several transforms are done on the data in order to generate the graph structures used by training.
        The processed dataset is automatically placed in a directory named processed, with the name of the processed file name property. If the processed file already exists in the correct directory, the processing step will be skipped.
        :return: Torch Geometric Dataset
        """
        # NB: coding for atom types
        types = {'H': 0, 'O': 1}

        S = np.load(self.split_file)
        if self.set_type == 'all':
            data_idx = np.concatenate((S['train_idx'], S['examine_idx'], S['test_idx'], S['val_idx'], S['reserve_idx'], S['add_idx']))
        else:
            data_idx = S[f'{self.set_type}_idx']
        data_list = []
        
        dbfile = osp.join(self.root, "raw", self.raw_file_names)
        assert osp.isfile(dbfile), f"Database file not found: {dbfile}"

        with connect(dbfile) as conn:
            center = True
            for i in range(len(data_idx)):
                idx_row = int(data_idx[i])
                row = conn.get(id=idx_row+1)
                name = ['energy', 'forces']
                mol = row.toatoms()
                y = torch.tensor(row.data[name[0]], dtype=torch.float) #potential energy
                f = torch.tensor(row.data[name[1]], dtype=torch.float) #gradients
                if center:
                    pos = mol.get_positions() - mol.get_center_of_mass()
                else:
                    pos = mol.get_positions()
                pos = torch.tensor(mol.get_positions(), dtype=torch.float) #coordinates
                size = int(pos.size(dim=0)/3)
                type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
                atomic_number = mol.get_atomic_numbers()
                z = torch.tensor(atomic_number, dtype=torch.long)
                x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(self.atom_types))

                data = Data(x=x, z=z, pos=pos, y=y, f=f, size=size, name=name, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                    # The graph edge_attr and edge_indices are created when the transforms are applied
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                        
                data_list.append(data)
                    
        torch.save(self.collate(data_list), self.processed_paths[0])
        return self.collate(data_list)

