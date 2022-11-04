"""
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor,
under Contract No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software
are reserved by DOE on behalf of the United States Government and the Contractor as provided in the Contract.
You are authorized to use this computer software for Governmental purposes but it is not to be released or
distributed to the public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  This notice including this sentence must appear on any
copies of this computer software.
"""

from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.db import connect
import pandas as pd
import numpy as np
import os.path as op
from utils.calc import TTMCalculator, SchnetCalculator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./data/models/TTM_nonminima.pt', type=str, help='Path to trained model.')
parser.add_argument('--database', required=True, type=str, help='Path to ASE database.')
parser.add_argument('--sample', required=True, type=int, help='Index of sample in ASE database.')
parser.add_argument('--seed', default=42, type=int, help='Random seed for repeated runs.')
parser.add_argument('--savedir', default='./results/md_runs', type=str, help='Directory to save results.')
parser.add_argument('--temp', default=300, type=float, help='Temperature of MD run.')
parser.add_argument('--step', default=0.1, type=float, help='Step size of MD run.')
parser.add_argument('--steps', default=1000, type=int, help='Number of steps in MD run.')
args = parser.parse_args()

####LOAD DATA####
db = connect(args.database)

n=args.sample
atoms_init=db[db.get(n).id].toatoms()
# apply small perturbation to coordinates to randomize
atoms_init.rattle(stdev=0.01, seed=args.seed)

####CALIBRATE#####
MaxwellBoltzmannDistribution(atoms_init, temperature_K=args.temp, force_temp=True)

####RUN NNP DYNAMICS####
db_tag = args.database.split('/')[-1].replace('.db','')
model_tag = args.model.split('/')[-1].replace('.pt','') 
traj_file = op.join(args.savedir, f"{db_tag}_row{n}_{model_tag}_{args.seed}.traj")
atoms = atoms_init.copy()

calc = SchnetCalculator(args.model, atoms)
atoms.calc = calc

dyn = NVTBerendsen(atoms, args.step * units.fs, temperature_K=args.temp, taut=0.5*1000*units.fs,
                   trajectory = traj_file)
dyn.run(args.steps)


print(f"NNP traj file saved to {traj_file}")


