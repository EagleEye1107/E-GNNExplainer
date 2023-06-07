# E-GraphSAGE Architecture

from dgl import from_networkx
import sklearn
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import numpy as np
from sklearn.metrics import confusion_matrix

import os
from sklearn.utils import shuffle

from dgl.data.utils import save_graphs

#constante
size_embedding = 152
nb_batch = 5

#Data
nbclasses =  2

# Accuracy --------------------------------------------------------------------
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()
# -----------------------------------------------------------------------------

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
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
            # Save edge_embeddings
            # nf = 'edge_embeddings'+str(i)+'.txt'
            # sourceFile = open(nf, 'w')
            # print(nfeats, file = sourceFile)
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
        # if(pr == True):
            # sourceFile = open(filename, 'w')
            # if pr:
                # print(v, file = sourceFile)
            # sourceFile.close()
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
    def forward(self, g, nfeats, efeats, eweight = None):
        if eweight != None:
            # apply eweight on the graph
            efe = []
            for i, x in enumerate(eweight):
                efe.append(list(th.Tensor.cpu(g.edata['h'][i][0]).detach().numpy() * th.Tensor.cpu(x).detach().numpy()))

            efe = th.FloatTensor(efe).cuda()
            efe = th.reshape(efe, (efe.shape[0], 1, efe.shape[1]))
            g.edata['h'] = efe = efe

        h = self.gnn(g, nfeats, efeats)
        # h = list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
        return self.pred(g, h)

# -------------------------------------------------------------------------------------------------------------------------------



# loading Graphs and Predictions
from dgl.data.utils import load_graphs
import numpy as np

Test_Graph = load_graphs("/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/G1_test.bin")
Test_Graph = Test_Graph[0][0]
Test_Graph_ab = load_graphs("/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/G1_test_ab.bin")
Test_Graph_ab = Test_Graph_ab[0][0]

Test_pred_Graph = np.loadtxt('/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/test_pred1.txt', dtype=int)
Test_pred_Graph_ab = np.loadtxt('/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/test_pred1_ab.txt', dtype=int)


actual1 = Test_Graph.edata['label']
actual1_ab = Test_Graph_ab.edata['label']


model1_test = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
model1_test.load_state_dict(th.load('/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/GNN.pt'))
model1_test.eval()

model1_test_ab = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
model1_test_ab.load_state_dict(th.load('/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/GNN_ab.pt'))
model1_test_ab.eval()


print("\\\\\\\\\\\\\\\\ NORMAL \\\\\\\\\\\\\\\\\\")

Test_Graph1 = Test_Graph.to('cuda:0')
node_features_test1 = Test_Graph1.ndata['feature']
edge_features_test1 = Test_Graph1.edata['h']

test_pred1 = model1_test(Test_Graph1, node_features_test1, edge_features_test1).cuda()
test_pred1 = test_pred1.argmax(1)
test_pred1 = th.Tensor.cpu(test_pred1).detach().numpy()

print('Metrics : ')
print("Accuracy : ", sklearn.metrics.accuracy_score(actual1, test_pred1))
print("Precision : ", sklearn.metrics.precision_score(actual1, test_pred1, labels = [0,1]))
print("Recall : ", sklearn.metrics.recall_score(actual1, test_pred1, labels = [0,1]))
print("f1_score : ", sklearn.metrics.f1_score(actual1, test_pred1, labels = [0,1]))


print("\\\\\\\\\\\\\\\\ ABLATION \\\\\\\\\\\\\\\\\\")

Test_Graph_ab1 = Test_Graph_ab.to('cuda:0')
node_features_test1_ab = Test_Graph_ab1.ndata['feature']
edge_features_test1_ab = Test_Graph_ab1.edata['h']

test_pred1_ab = model1_test_ab(Test_Graph_ab1, node_features_test1_ab, edge_features_test1_ab).cuda()
test_pred1_ab = test_pred1_ab.argmax(1)
test_pred1_ab = th.Tensor.cpu(test_pred1_ab).detach().numpy()

print('Metrics : ')
print("Accuracy : ", sklearn.metrics.accuracy_score(actual1_ab, test_pred1_ab))
print("Precision : ", sklearn.metrics.precision_score(actual1_ab, test_pred1_ab, labels = [0,1]))
print("Recall : ", sklearn.metrics.recall_score(actual1_ab, test_pred1_ab, labels = [0,1]))
print("f1_score : ", sklearn.metrics.f1_score(actual1_ab, test_pred1_ab, labels = [0,1]))


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# Explain the 3% difference
print("Explain the 3% difference")


# Local Explanation ***********************************************************************
from math import sqrt
from tqdm import tqdm
from dgl import EID, NID, khop_out_subgraph
import torch.nn as nn
import torch as th


# init mask
def init_masks(graph, efeat):
    # efeat.size() = torch.Size([nb_edges, 1, 76])
    efeat_size = efeat.size()[2]
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()

    device = efeat.device

    std = 0.1
    # feat_mask = [[f1, f2, .... fn]] / n = nb_features
    efeat_mask = nn.Parameter(th.randn(1, efeat_size, device=device) * std)

    std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
    # edge_mask = [e1, e2, .... em] / m = nb_edges
    edge_mask = nn.Parameter(th.randn(num_edges, device=device) * std)

    # print("efeat_mask : ", efeat_mask)
    # print("edge_mask : ", edge_mask)

    return efeat_mask, edge_mask


# Regularization loss
def loss_regularize(loss, feat_mask, edge_mask):
    # epsilon for numerical stability
    eps = 1e-15
    # From self GNNExplainer self
    alpha1 = 0.005,
    alpha2 = 1.0
    beta1 = 1.0
    beta2 = 0.1

    edge_mask = edge_mask.sigmoid()
    # Edge mask sparsity regularization
    loss = loss + th.from_numpy(alpha1 * th.Tensor.cpu(th.sum(edge_mask)).detach().numpy()).cuda()
    # Edge mask entropy regularization
    ent = -edge_mask * th.log(edge_mask + eps) - (
        1 - edge_mask
    ) * th.log(1 - edge_mask + eps)
    loss = loss + alpha2 * ent.mean()

    feat_mask = feat_mask.sigmoid()
    # Feature mask sparsity regularization
    loss = loss + beta1 * th.mean(feat_mask)
    # Feature mask entropy regularization
    ent = -feat_mask * th.log(feat_mask + eps) - (
        1 - feat_mask
    ) * th.log(1 - feat_mask + eps)
    loss = loss + beta2 * ent.mean()

    return loss



# Edge mask
def explain_edges(model, edge_id, graph, node_feat, edge_feat, **kwargs):
    model = model.to(graph.device)
    model.eval()

    #print(graph.edges())

    # Extract source node-centered k-hop subgraph from the edge_id and its associated node and edge features.
    num_hops = 3
    source_node = th.Tensor.cpu(graph.edges()[0][edge_id]).detach().numpy()
    #print("source_node : ", source_node)
    edge_h = graph.edata['h'][edge_id]
    sg, inverse_indices = khop_out_subgraph(graph, source_node, num_hops)
    #print("new_node_indice : ", inverse_indices)

    #print(sg.edges())
    #print(edge_h)
    #print(sg.edata['h'])

    for indx, nd_id in enumerate(sg.edges()[0]):
        if inverse_indices == nd_id :
            if (sg.edata['h'][indx][0] == edge_h[0]).all() :
                # print("edge index is : ", indx)
                edge_indice = indx
                break
    
    #print("new_edge_indice : ", edge_indice)

    # EID = NID = _ID
    # tensor([0, 1, 2, 4]) : nodes and edges ids
    sg_edges = sg.edata[EID].long()
    sg_nodes = sg.ndata[NID].long()

    #print("+++++++++++++++++++++++")
    #print("sg : ", sg)
    #print("sg_edges : ", sg_edges) # edges ids in graph.edges()
    #print("sg_nodes : ", sg_nodes) # nodes ids in graph.nodes()

    #print()
    edge_feat = edge_feat[sg_edges]
    node_feat = node_feat[sg_nodes]

    #print("edge_feat : ", edge_feat)
    #print("node_feat : ", node_feat)
    #print("+++++++++++++++++++++++")
    
    
    # Get the initial prediction.
    #print("Get the initial prediction :")
    with th.no_grad():
        # logits = model(g = sg, nfeats = node_feat, efeats = edge_feat, **kwargs)
        logits = model(g = sg, nfeats = node_feat, efeats = edge_feat)
        pred_label = logits.argmax(dim=-1)
        # pred_label1 = logits.argmax(1)

    #print("pred_label : ", pred_label)
    # print(pred_label1)

    #
    efeat_mask, edge_mask = init_masks(sg, edge_feat)

    params = [edge_mask]
    optimizer = th.optim.Adam(params, lr = 0.01)

    # num_epochs = 300
    #print("***********************************")
    #print("initial masks : ")
    #print("efeat_mask : ", efeat_mask)
    #print("edge_mask : ", edge_mask)
    #print("***********************************")
    
    
    from sklearn.utils import class_weight
    # class_weights2 = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(sg.edata['label'].cpu().numpy()), y = sg.edata['label'].cpu().numpy())
    # class_weights2 = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.array([0, 1]), y = sg.edata['label'].cpu().numpy())
    # class_weights2 = th.FloatTensor(class_weights2).cuda()
    # criterion2 = nn.CrossEntropyLoss(weight = class_weights2)
    criterion2 = nn.CrossEntropyLoss()
    train_mask2 = th.ones(len(sg.edata['h']), dtype=th.bool)
    import datetime
    
    #print(f'explanation starts at {datetime.datetime.now()}')
    #print("nb edges : ", sg.num_edges())
    #print("nb nodes : ", sg.num_nodes())
    
    
    for epoch in range(1, 300):
        optimizer.zero_grad()
        # Edge mask
        logits = model(g = sg, nfeats = node_feat, efeats = edge_feat, eweight=edge_mask.sigmoid()).cuda()
        # logits = model(g = sg, nfeats = node_feat, efeats = h)
        # pred_label = tensor([0, 0, 0,  ..., 0, 1, 0], device='cuda:0')
        # logits = tensor([[ 0.0059,  0.0517], [-0.0075,  0.0101], ..., device='cuda:0', grad_fn=<IndexBackward0>)
        # log_probs = logits.log_softmax(dim=-1)
        # loss = -log_probs[edge_indice, pred_label[edge_indice]]
        loss11 = criterion2(logits[train_mask2], pred_label[train_mask2])
        loss = loss_regularize(loss11, efeat_mask, edge_mask)
        # loss = loss_regularize(loss, efeat_mask, edge_mask)
        
        #if epoch % 100 == 0:
            #print("+++++++++++++++")
            #print(f'epoch number {epoch}, CrossEntropy_Loss = {loss11}, final_loss = {loss}, time = {datetime.datetime.now()}')
            #print("edge_mask : ", edge_mask.detach().sigmoid())
        
        loss.backward()
        optimizer.step()

    #print("final results before sigmoid : ")
    #print("edge_mask : ", edge_mask)
    #print("***********************************")

    edge_mask = edge_mask.detach().sigmoid()

    return edge_indice, sg, edge_mask, loss.item()


Test_Graph1 = Test_Graph.to('cuda:0')
node_features_test1 = Test_Graph1.ndata['feature']
edge_features_test1 = Test_Graph1.edata['h']

indx_edges_to_explain = pd.read_csv('/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/Edges_to_explain.csv', encoding="ISO-8859–1", dtype = str)
indx_edges_to_explain = indx_edges_to_explain.apply(pd.to_numeric)
print(indx_edges_to_explain.dtypes)
print(indx_edges_to_explain)

# results_df = pd.DataFrame(columns = ['edge_indx', 'label', 'edge_indice', 'sg', 'edge_mask', 'loss'])
results_df = pd.DataFrame(columns = ['edge_indx', 'label', 'loss'])


# Explanations
print("nb edges to explain =", len(indx_edges_to_explain['Edge_indx']))
for i, x in enumerate(indx_edges_to_explain['Edge_indx']):
    if (i % 100) == 0 :
        print(f"{i}th edge")
    edge_indice, sub_graph, edge_mask, loss = explain_edges(model1_test, x, Test_Graph1, node_features_test1, edge_features_test1)
    results_df.loc[-1] = [x, indx_edges_to_explain['label'][i], loss]  # adding a row
    results_df.index = results_df.index + 1  # shifting index


results_df.to_csv(f'/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/jupyter_notebooks/XAI/GNNExplainer/Models/Final_results.csv', sep=',', index = False)
