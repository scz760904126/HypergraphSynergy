#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: Scz 
# @Time:  2022/3/31 16:41
import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
from utils import get_MACCS
from drug_util import drug_feature_extract


def getData(dataset):
    if dataset == 'ONEIL':
        drug_smiles_file = '../Data/ONEIL-COSMIC/drug_smiles.csv'
        cline_feature_file = '../Data/ONEIL-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../Data/ONEIL-COSMIC/drug_synergy.csv'
    else:
        drug_smiles_file = '../Data/ALMANAC-COSMIC/drug_smiles.csv'
        cline_feature_file = '../Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../Data/ALMANAC-COSMIC/drug_synergy.csv'
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data = pd.DataFrame()
    drug_smiles_fea = []
    featurizer = dc.feat.ConvMolFeaturizer()
    for tup in zip(drug['pubchemid'], drug['isosmiles']):
        mol = Chem.MolFromSmiles(tup[1])
        mol_f = featurizer.featurize(mol)
        drug_data[str(tup[0])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
        drug_smiles_fea.append(get_MACCS(tup[1]))
    drug_num = len(drug_data.keys())
    d_map = dict(zip(drug_data.keys(), range(drug_num)))
    drug_fea = drug_feature_extract(drug_data)
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_num = len(gene_data.index)
    c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    cline_fea = np.array(gene_data, dtype='float32')
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
               synergy_load.iterrows() if (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and
                                           str(row[2]) in gene_data.index)]
    return cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy