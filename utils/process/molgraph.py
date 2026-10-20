"""SMILES -> typed molecular graph for the R-GCN branch (chemistry instance of BDCA).

The text branch reads the SMILES string with a chemical language model; this module builds the
graph of the same molecule:

* nodes = heavy atoms (hydrogens implicit, as in OGB / MoleculeNet practice);
* node features ``x`` = concatenated one-hot encodings of the nine OGB atom features
  (atomic number, chirality tag, degree, formal charge, number of Hs, radical electrons,
  hybridisation, aromaticity, ring membership). The vocabularies follow
  ``ogb.utils.features`` (Hu et al. 2020, Open Graph Benchmark) so numbers are comparable with
  OGB-featurised GNN baselines; the last entry of each list is "misc" for unseen values;
* ``node_label`` = the atomic-number index (embedded by the model, like the Joern node label);
* edges = bonds, added in both directions with the SAME relation, relation = bond type
  (single, double, triple, aromatic, misc) -> ``NUM_BOND_RELATIONS`` = 5;
* a molecule without bonds (single heavy atom, salts split into one fragment of one atom)
  gets self loops of relation 0, as the code graphs do.

Nothing here is learned or random: the same SMILES always yields the same graph.

The vocabularies and accessors are OGB's. Built from a canonical SMILES (as the pipeline does), the
chirality tag CW / CCW is relative to RDKit's neighbour order of THAT string, so it can be the
opposite of OGB's shipped node features for re-ordered stereocentres (BBBP 455, BACE 277 molecules);
all other atom features and every bond type equal OGB's.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

try:  # optional dependency: only needed for the chemistry datasets
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover - reported when used
    Chem = None

# --- OGB allowable features (ogb/utils/features.py) -----------------------------------------
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER", "misc"],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "radical_electrons": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}
ATOM_FEATURE_DIMS = [len(v) for v in ATOM_FEATURES.values()]
ATOM_FEAT_DIM = sum(ATOM_FEATURE_DIMS)                     # 174
NUM_ATOM_LABELS = len(ATOM_FEATURES["atomic_num"]) + 1     # 119 + 1 (index 0 unused: atomic numbers start at 1)
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
NUM_BOND_RELATIONS = len(BOND_TYPES)                       # 5


def _index(values, v):
    try:
        return values.index(v)
    except ValueError:
        return len(values) - 1                              # "misc"


def atom_feature_indices(atom) -> list[int]:
    return [
        _index(ATOM_FEATURES["atomic_num"], atom.GetAtomicNum()),
        _index(ATOM_FEATURES["chirality"], str(atom.GetChiralTag())),
        _index(ATOM_FEATURES["degree"], atom.GetTotalDegree()),
        _index(ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()),
        _index(ATOM_FEATURES["num_hs"], atom.GetTotalNumHs()),
        _index(ATOM_FEATURES["radical_electrons"], atom.GetNumRadicalElectrons()),
        _index(ATOM_FEATURES["hybridization"], str(atom.GetHybridization())),
        _index(ATOM_FEATURES["is_aromatic"], atom.GetIsAromatic()),
        _index(ATOM_FEATURES["is_in_ring"], atom.IsInRing()),
    ]


def _require_rdkit():
    if Chem is None:
        raise ImportError("the chemistry datasets need RDKit: pip install rdkit")


def mol_from_smiles(smiles: str):
    _require_rdkit()
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None


def canonical_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def smiles_to_data(smiles: str, target: int) -> Data | None:
    """PyG Data(x [N, 174] float one-hot, edge_index [2, E], edge_type [E], node_label [N], y [1]),
    or None when RDKit cannot parse the SMILES or the molecule has no heavy atom."""
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    n = mol.GetNumAtoms()
    idx = torch.tensor([atom_feature_indices(a) for a in mol.GetAtoms()], dtype=torch.long)    # [N, 9]
    offsets = torch.tensor([0] + ATOM_FEATURE_DIMS[:-1]).cumsum(0)
    x = torch.zeros(n, ATOM_FEAT_DIM)
    x.scatter_(1, idx + offsets, 1.0)
    node_label = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() < NUM_ATOM_LABELS else 0
                               for a in mol.GetAtoms()], dtype=torch.long)
    src, dst, et = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        r = _index(BOND_TYPES, str(b.GetBondType()))
        src += [i, j]; dst += [j, i]; et += [r, r]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(et, dtype=torch.long)
    else:                                                    # no bond: self loops, relation 0
        edge_index = torch.arange(n).repeat(2, 1)
        edge_type = torch.zeros(n, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, node_label=node_label,
                y=torch.tensor([int(target)], dtype=torch.long))
    data.num_nodes = n
    return data
