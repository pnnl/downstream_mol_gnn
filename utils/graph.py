"""
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor,
under Contract No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software
are reserved by DOE on behalf of the United States Government and the Contractor as provided in the Contract.
You are authorized to use this computer software for Governmental purposes but it is not to be released or
distributed to the public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  This notice including this sentence must appear on any
copies of this computer software.
"""

from ase import data
import networkx as nx
import numpy as np


def infer_water_cluster_bonds(atoms):
    """
    Infers the covalent and hydrogen bonds between oxygen and hydrogen atoms in a water cluster.
    Definition of a hydrogen bond obtained from https://aip.scitation.org/doi/10.1063/1.2742385
    Args:
        atoms (ase.Atoms): ASE atoms structure of the water cluster. Atoms list must be ordered
            such that the two covalently bound hydrogens directly follow their oxygen.
    Returns:
        cov_bonds ([(str, str, 'covalent')]): List of all covalent bonds
        h_bonds [(str, str, 'hydrogen')]: List of all hydrogen bonds
    """

    # Make sure the atoms are in the right order
    z = atoms.get_atomic_numbers()
    assert z[:3].tolist() == [8, 1, 1], "Atom list not in (O, H, H) format"
    coords = atoms.positions

    # Get the covalent bonds
    #  Note: Assumes that each O is followed by 2 covalently-bonded H atoms
    cov_bonds = [(i, i + 1, 'covalent') for i in range(0, len(atoms), 3)]
    cov_bonds.extend([(i, i + 2, 'covalent') for i in range(0, len(atoms), 3)])

    # Get the hydrogen bonds
    #  Start by getting the normal to each water molecule
    q_1_2 = []
    for i in range(0, len(atoms), 3):
        h1 = coords[i + 1, :]
        h2 = coords[i + 2, :]
        o = coords[i, :]
        q_1_2.append([h1 - o, h2 - o])
    v_list = [np.cross(q1, q2) for (q1, q2) in q_1_2]

    #  Determine which (O, H) pairs are bonded
    h_bonds = []
    for idx, v in enumerate(v_list):  # Loop over each water molecule
        for index, both_roh in enumerate(q_1_2):  # Loop over each hydrogen
            for h_index, roh in enumerate(both_roh):
                # Get the index of the H and O atoms being bonded
                indexO = 3 * idx
                indexH = 3 * index + h_index + 1

                # Get the coordinates of the two atoms
                h_hbond = coords[indexH, :]
                o_hbond = coords[indexO, :]

                # Compute whether they are bonded
                dist = np.linalg.norm(h_hbond - o_hbond)
                if (dist > 1) & (dist < 2.8):
                    angle = np.arccos(np.dot(roh, v) / (np.linalg.norm(roh) * np.linalg.norm(v))) * (180.0 / np.pi)
                    if angle > 90.0:
                        angle = 180.0 - angle
                    N = np.exp(-np.linalg.norm(dist) / 0.343) * (7.1 - (0.05 * angle) + (0.00021 * (angle ** 2)))
                    if N >= 0.0085:
                        h_bonds.append((indexO, indexH, 'hydrogen'))

    return cov_bonds, h_bonds


def create_graph(atoms):
    """
    Given a ASE atoms object, this function returns a graph structure with following properties.
        1) Each graph has two graph-level attributes: actual_energy and predicted_energy
        2) Each node represents an atom and has two attributes: label ('O'/'H' for oxygen and hydrogen) and 3-dimensional
           coordinates.
        3) Each edge represents a bond between two atoms and has two attributes: label (covalent or hydrogen) and distance.
    Args:
        atoms (Atoms): ASE atoms object
    Returns:
        (nx.Graph) Networkx representation of the water cluster
    """

    # Compute the bonds
    cov_bonds, h_bonds = infer_water_cluster_bonds(atoms)

    # Add nodes to the graph
    graph = nx.Graph()
    for i, (coord, Z) in enumerate(zip(atoms.positions, atoms.get_atomic_numbers())):
        graph.add_node(i, label=data.chemical_symbols[Z], coords=coord)

    # Add the edges
    edges = cov_bonds + h_bonds
    for a1, a2, btype in edges:
        distance = np.linalg.norm(atoms.positions[a1, :] - atoms.positions[a2, :])
        graph.add_edge(a1, a2, label=btype, weight=distance)
    return graph


def coarsen_graph(in_graph: nx.Graph) -> nx.DiGraph:
    """Create a graph with only one node per water molecule
    Args:
        in_graph: Input graph, which contains both hydrogens and oxygen
    Returns:
         A directed graph with only the oxygen atoms as nodes. Nodes are identified
         as whether that O atoms is part of a water molecule that donates a hydrogen
         bond to another molecule or whether it receives a hydrogen bond
    """

    # Initialize the output graph
    output = nx.DiGraph()

    # Collect information from the previous graph
    bonds = []
    for node, node_data in in_graph.nodes(data=True):
        if node_data['label'] == 'O':
            # If it is an oxygen, make it a node in the new graph
            output.add_node(node // 3, **node_data)  # Make the count from [0, N)
        elif node_data['label'] == 'H':
            # Check if this H participates in H bonding
            donor = acceptor = None  # Stores the donor and acceptor oxygen id
            for neigh in in_graph.neighbors(node):
                neigh_info = in_graph.get_edge_data(node, neigh)
                bond_type = neigh_info['label']
                if bond_type == 'covalent':
                    donor = neigh // 3, neigh_info['weight']
                else:
                    acceptor = neigh // 3, neigh_info['weight']
            if not (donor is None or acceptor is None):
                bonds.append((donor, acceptor))  # Store as donor->acceptor
        else:
            raise ValueError(f'Unrecognized type: {node_data["label"]}')

    # Assign bonds to each water molecule
    for (d, w_d), (a, w_a) in bonds:
        # Add the edges
        output.add_edge(d, a, label='donate', weight=w_d+w_a)
        #output.add_edge(a, d, label='accept', weight=w_d+w_a)

    return output
