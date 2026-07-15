"""
featurizer.py — Convert SMILES strings to molecular graph (PyG Data) or tokenized text.

Uses RDKit to parse SMILES and extract atom-level / bond-level features,
then packages them into torch_geometric.data.Data objects.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data

# ── Atom feature vocabulary ───────────────────────────────────────────────────
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53, 1]  # C,N,O,F,P,S,Cl,Br,I,H
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Dimensions
NUM_ATOM_FEATURES = len(ATOM_TYPES) + 1 + 6 + len(HYBRIDIZATIONS) + 1 + 1 + 5 + 1
#                   atom_type(10+1)  degree(6) hybridization(5)     arom  ring  numH(5) charge_sign
# = 11 + 6 + 5 + 1 + 1 + 5 + 1 = 30

NUM_BOND_FEATURES = len(BOND_TYPES) + 1 + 1 + 1  # bond_type(4+1) + conjugated(1) + in_ring(1) = 7


def _one_hot(value, vocabulary):
    """One-hot encode *value* against *vocabulary*. Unknown → all zeros."""
    vec = [0] * (len(vocabulary) + 1)  # +1 for "other"
    if value in vocabulary:
        vec[vocabulary.index(value)] = 1
    else:
        vec[-1] = 1  # "other" flag
    return vec


def _atom_features(atom):
    """Return a numerical feature vector for a single RDKit atom."""
    features = []
    # Atom type one-hot
    features += _one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    # Degree one-hot (0-5)
    features += _one_hot(atom.GetTotalDegree(), list(range(6)))
    # Hybridization one-hot
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    # Is aromatic
    features.append(int(atom.GetIsAromatic()))
    # In ring
    features.append(int(atom.IsInRing()))
    # Number of Hs one-hot (0-4)
    features += _one_hot(atom.GetTotalNumHs(), list(range(5)))
    # Formal charge sign (-1, 0, +1 mapped to float)
    fc = atom.GetFormalCharge()
    features.append(float(np.sign(fc)))
    return features


def _bond_features(bond):
    """Return a numerical feature vector for a single RDKit bond."""
    features = []
    features += _one_hot(bond.GetBondType(), BOND_TYPES)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    return features


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES is invalid or the molecule has no atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Node features ─────────────────────────────────────────────────────
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(_atom_features(atom))
    if len(atom_feats) == 0:
        return None
    x = torch.tensor(atom_feats, dtype=torch.float)

    # ── Edge index + edge features ────────────────────────────────────────
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        # Undirected: add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Molecule with no bonds (single atom)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_BOND_FEATURES), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def smiles_to_2d_coords(smiles: str):
    """
    Return 2D atom coordinates + bond info for frontend molecule rendering.

    Returns a dict with:
      - atoms: list of {idx, symbol, x, y}
      - bonds: list of {begin, end, type}
    or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from rdkit.Chem import AllChem
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
        })

    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "begin": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(),
            "type": str(bond.GetBondType()).split(".")[-1],  # e.g. "SINGLE"
        })

    return {"atoms": atoms, "bonds": bonds}


def get_num_atom_features() -> int:
    """Return the dimensionality of atom feature vectors."""
    # Compute dynamically from a dummy molecule
    mol = Chem.MolFromSmiles("C")
    return len(_atom_features(mol.GetAtomWithIdx(0)))


def get_num_bond_features() -> int:
    """Return the dimensionality of bond feature vectors."""
    mol = Chem.MolFromSmiles("CC")
    return len(_bond_features(mol.GetBondWithIdx(0)))
