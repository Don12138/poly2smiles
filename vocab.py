import logging
import os.path

import networkx as nx
import numpy as np
import re
import selfies as sf
import sys
import time
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from utils.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse, get_bond_features
from utils.rxn_graphs import RxnGraph
from utils.data_utils import get_graph_features_from_smi, load_vocab, make_vocab, \
    tokenize_selfies_from_smiles, tokenize_smiles



train_src = "/home/chenlidong/poly2SMILES/data/poly/src-train.txt"
train_tgt = "/home/chenlidong/poly2SMILES/data/poly/tgt-train.txt"
val_src = "/home/chenlidong/poly2SMILES/data/poly/src-val.txt"
val_tgt = "/home/chenlidong/poly2SMILES/data/poly/tgt-val.txt"
test_src = "/home/chenlidong/poly2SMILES/data/poly/src-test.txt"
test_tgt = "/home/chenlidong/poly2SMILES/data/poly/tgt-test.txt"
fns = {
    "train": [(train_src, train_tgt)],
    "val": [(val_src, val_tgt)],
    "test": [(test_src, test_tgt)]
}
make_vocab(
    fns=fns,
    vocab_file="./vocab2.txt",
    tokenized=True
)