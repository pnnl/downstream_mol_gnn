# Reducing Down(stream)time: Pretraining Molecular GNNs using Heterogeneous AI Accelerators 

## Conda Environment

#### Using pip with conda
Not all packages are available with conda. To correctly direct a pip install in a conda environment, first `conda install pip`. Pip will install in your anaconda (or conda or miniconda) directory under the name of your environment (something like `/anaconda/envs/env_name/`). In all subsequent pip installs, replace `pip` with `/anaconda/envs/env_name/bin/pip`.

#### Pytorch 1.9.0 with cuda 11.1
This installation was used for training across NVIDIA P100s and RTX 2080 Ti GPUs.
```
conda install pytorch==1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm gdown
```

Note that it may be necessary to downgrade setuptools if tensorboard throws an error:
```
pip install setuptools==59.5.0
```

#### Pytorch 1.12.0 with cuda 11.3
This installation was used for training across NVIDIA A100s.
```
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm gdown
```
Note that installing `torch-spine-conv` will likely produce a GLIBC error. It is safe to `pip uninstall torch-spine-conv` if the error occurs.
