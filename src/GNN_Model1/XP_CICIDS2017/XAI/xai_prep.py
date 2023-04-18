''''
    This python file refers to the basic preparation to load the data and the model
    to do so lets try it as the original model
'''


from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import numpy as np

#constante
size_embedding = 152

# to print
pr = True
filename = './models/M1_weights.txt'

# ------------------------------------------ Model Architecture -----------------------------------------------------------------

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        x = th.cat([edges.src['h'], edges.data['h']], 2)
        y = self.W_msg(x)
        return {'m': y}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # Line 4 of algorithm 1 : update all because we are using a full neighborhood sampling and not a k-hop neigh sampling
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            # Line 5 of algorithm 1
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, size_embedding, activation))
        self.layers.append(SAGELayer(size_embedding, edim, size_embedding, activation)) ##
        self.layers.append(SAGELayer(size_embedding, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        
        for i, layer in enumerate(self.layers):
            #nf = 'weights'+str(i)+'.txt'
            #sourceFile = open(nf, 'w')
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)
        # Return a list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
    
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        v = th.cat([h_u, h_v], 1)
        if(pr == True):
            sourceFile = open(filename, 'w')
            if pr:
                print(v, file = sourceFile)
            sourceFile.close()
        score = self.W(v)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            # Update the features of the specified edges by the provided function
            # DGLGraph.apply_edges(func, edges='__ALL__', etype=None, inplace=False)
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, nbclasses)
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        # h = list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
        return self.pred(g, h)

# -------------------------------------------------------------------------------------------------------------------------------




# Loading data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
xai_datafile = "./input/Dataset/XAI/XAI_Test.csv"
gen_xai_testset = pd.read_csv(xai_datafile, encoding="ISO-8859–1", dtype = str)

labels_column = gen_xai_testset.label

print(gen_xai_testset["label"].value_counts())

# Create our Multigraph
XAI_G1 = nx.from_pandas_edgelist(gen_xai_testset, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
XAI_G1 = XAI_G1.to_directed()
XAI_G1 = from_networkx(XAI_G1, edge_attrs=['h','label'] )

XAI_G1.ndata['h'] = th.ones(XAI_G1.num_nodes(), XAI_G1.edata['h'].shape[1])
XAI_G1.edata['train_mask'] = th.ones(len(XAI_G1.edata['h']), dtype=th.bool)
# Reshape both tensor lists to a single value in each element for both axis
XAI_G1.ndata['h'] = th.reshape(XAI_G1.ndata['h'], (XAI_G1.ndata['h'].shape[0], 1, XAI_G1.ndata['h'].shape[1]))
XAI_G1.edata['h'] = th.reshape(XAI_G1.edata['h'], (XAI_G1.edata['h'].shape[0], 1, XAI_G1.edata['h'].shape[1]))

XAI_G1 = XAI_G1.to('cuda:0')

node_features1 = XAI_G1.ndata['h']
edge_features1 = XAI_G1.edata['h']
edge_label1 = XAI_G1.edata['label']
train_mask1 = XAI_G1.edata['train_mask']

print(XAI_G1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# Loading the model ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nbclasses =  2

# Model *******************************************************************************************
# G1.ndata['h'].shape[2] = sizeh = 76 dans ANIDS
# model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
# model1 = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
# model1.load_state_dict(th.load("./models/Model1/model1.pt"))
# model1.eval()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# Testing everything ++++++++++++++++++++++++++

# pred1 = model1(XAI_G1, node_features1, edge_features1).cuda()
# print('Train metrics :')
# print("Accuracy : ", sklearn.metrics.accuracy_score(edge_label1, pred1))
# print("Precision : ", sklearn.metrics.precision_score(edge_label1, pred1, labels = [0,1]))
# print("Recall : ", sklearn.metrics.recall_score(edge_label1, pred1, labels = [0,1]))
# print("f1_score : ", sklearn.metrics.f1_score(edge_label1, pred1, labels=[0,1]))

# +++++++++++++++++++++++++++++++++++++++++++++