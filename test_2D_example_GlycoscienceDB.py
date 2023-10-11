import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import torch.nn as nn
from dgl import AddSelfLoop

from model_2d.NMR_gcn import NMR_GCN

from train_evaluate_2d import NMR_prediction

from preprocess.create_node_embedding.node_embeddings_glycoscience import create_node_embeddings

from preprocess.build_adjaency_matrix.create_adjaency_matrix_glycoscience import build_adjacency_matrix

from preprocess.create_graph.create_graph_data_glycosciencedb import create_graph

from dgl import save_graphs, load_graphs

atom_dim = 256

residual_dim = 128

mono_dim = 64
ab_dim = 64
dl_dim = 64
pf_dim = 64

seed=9721



C = create_node_embeddings(seed=seed)

df_atom_embedding, df_residual_embedding, df_monosaccharide_embedding, df_ab_embedding, df_dl_embedding, df_pf_embedding =\
C.create_all_embeddings(atom_dim=atom_dim, residual_dim=residual_dim, mono_dim=mono_dim,
                       ab_dim=ab_dim, dl_dim=dl_dim, pf_dim=pf_dim)


df_atom_embedding.to_csv(C.out_atom_embed, index=False)
df_residual_embedding.to_csv(C.out_residual_embed, index=False)
df_monosaccharide_embedding.to_csv(C.out_monosaccharide_embed, index=False)

df_ab_embedding.to_csv(C.out_bound_ab, index = False)
df_dl_embedding.to_csv(C.out_bound_dl, index = False)
df_pf_embedding.to_csv(C.out_carbon_pf, index = False)

B = build_adjacency_matrix()
B.calculate_all_matrix()

num_test = 60 # 299 * 20%
Create = create_graph(num_test=num_test, seed=seed)
g, test_index = Create.create_all_graph()
num_epoch = 1000
lr = 1e-2

in_size = atom_dim + mono_dim + ab_dim + dl_dim + pf_dim


hidden_size_1 = int(in_size / 2)
hidden_size_2 = 256
hidden_size_3 = 128
hidden_size_4 = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = g.to(device)
features = g.ndata["feat"]
labels = g.ndata["shift_value"]
# masks = g.ndata['train_mask'], g.ndata['test_mask']


# masks = g.ndata['train_hydrogen_mask'], g.ndata['test_hydrogen_mask']
masks = g.ndata['train_carbon_mask'], g.ndata['test_carbon_mask']


print(features.dtype)
print(labels.dtype)
# model = NMR_GCN(in_size=576, hid_size=[256, 128, 64, 32], out_size=1).to(device)
model = NMR_GCN(in_size=in_size, hid_size=[hidden_size_1, hidden_size_2,
                                       hidden_size_3, hidden_size_4], out_size=1).to(device)
# model training

# NMR_prediction = NMR_prediction(results_dir='experimental_data_combined/graph_combined/experimental_results_all_hydrogen.csv',
#                                 model_dir='experimental_data_combined/graph_combined/Model_hydrogen_experiment_training.pt',
#                                num_epoch = num_epoch,
#                                lr = lr)


NMR_prediction = NMR_prediction(results_dir='glycosciencedb/results/training_carbon.csv',
                                results_dir_test = 'glycosciencedb/results/testing_carbon.csv',
                                model_dir='glycosciencedb/results/Model_Godess_carbon.pt',
                               num_epoch = num_epoch,
                               lr = lr)


print("Training...")
NMR_prediction.train(g, features, labels, masks, model)

# test the model
print("Testing...")
saved_model = NMR_GCN(in_size=in_size, hid_size=[hidden_size_1, hidden_size_2,
                                                 hidden_size_3, hidden_size_4], out_size=1).to(device)
saved_model.load_state_dict(torch.load(NMR_prediction.model_dir))

# acc = NMR_prediction.evaluate(g, features, labels, masks[0], saved_model, print_out=True)
acc1 = NMR_prediction.evaluate(g, features, labels, masks[0], saved_model, save_train=True, save_test = False)

acc2 = NMR_prediction.evaluate(g, features, labels, masks[1], saved_model, save_train=False, save_test = True)


print("train RMSE {:.4f}".format(acc1))
print("test RMSE {:.4f}".format(acc2))
