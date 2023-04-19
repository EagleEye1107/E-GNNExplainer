''''
    This python file refers to the basic preparation to load the data and the model
    to do so lets try it as the original model
'''


from dgl import from_networkx
import sklearn
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import numpy as np

import sklearn.metrics

import shap
import matplotlib.pyplot as plt

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

class GPreprocessing():
    def __init__(self):
        self.encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
        self.scaler1 = StandardScaler()
        super().__init__()
    
    def train(self, data1):
        # Preprocessing and creation of the h attribute
        label1 = data1['label']
        self.encoder1.fit(data1, label1)
        data1 = self.encoder1.transform(data1)
        # scaler (normalization)
        # Manipulate flow content (all columns except : label, Source IP & Destination IP)
        cols_to_norm1 = list(set(list(data1.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
        data1[cols_to_norm1] = self.scaler1.fit_transform(data1[cols_to_norm1])
        ## Create the h attribute that will contain the content of our flows
        data1['h'] = data1[ cols_to_norm1 ].values.tolist()
        # We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
        data1.drop(columns = cols_to_norm1, inplace = True)
        # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
        columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
        data1 = data1.reindex(columns=columns_titles)

        # Graph construction #################################################
        G1 = nx.from_pandas_edgelist(data1, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
        G1 = G1.to_directed()
        G1 = from_networkx(G1,edge_attrs=['h','label'] )
        G1.ndata['h'] = th.ones(G1.num_nodes(), G1.edata['h'].shape[1])
        G1.edata['train_mask'] = th.ones(len(G1.edata['h']), dtype=th.bool)
        G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
        G1.edata['h'] = th.reshape(G1.edata['h'], (G1.edata['h'].shape[0], 1, G1.edata['h'].shape[1]))
        return G1
    
    def test(self, data1):
        data1 = self.encoder1.transform(data1)
        cols_to_norm1 = list(set(list(data1.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
        data1[cols_to_norm1] = self.scaler1.transform(data1[cols_to_norm1])
        data1['h'] = data1[ cols_to_norm1 ].values.tolist()
        data1.drop(columns = cols_to_norm1, inplace = True)
        columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
        data1 = data1.reindex(columns=columns_titles)

        # Graph construction #################################################
        G1_test = nx.from_pandas_edgelist(data1, " Source IP", " Destination IP", ['h','label'],create_using=nx.MultiGraph())
        G1_test = G1_test.to_directed()
        G1_test = from_networkx(G1_test,edge_attrs=['h','label'] )
        # G1.ndata['h'].shape[2] = sizeh = 76
        G1_test.ndata['feature'] = th.ones(G1_test.num_nodes(), 76)
        G1_test.ndata['feature'] = th.reshape(G1_test.ndata['feature'], (G1_test.ndata['feature'].shape[0], 1, G1_test.ndata['feature'].shape[1]))
        G1_test.edata['h'] = th.reshape(G1_test.edata['h'], (G1_test.edata['h'].shape[0], 1, G1_test.edata['h'].shape[1]))
        return G1_test


class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.preprocessing = GPreprocessing()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, nbclasses)
    
    def train(self, data1, epochs):
        G1 = self.preprocessing.train(data1)
        G1 = G1.to('cuda:0')
        from sklearn.utils import class_weight
        class_weights1 = class_weight.compute_class_weight(class_weight = 'balanced',
                                                        classes = np.unique(G1.edata['label'].cpu().numpy()),
                                                        y = G1.edata['label'].cpu().numpy())
        class_weights1 = th.FloatTensor(class_weights1).cuda()
        criterion1 = nn.CrossEntropyLoss(weight = class_weights1)

        nfeats = G1.ndata['h']
        efeats = G1.edata['h']

        edge_label1 = G1.edata['label']
        train_mask1 = G1.edata['train_mask']

        for epoch in range(1, epochs):
            h = self.gnn(G1, nfeats, efeats).cuda()
            pred1 = self.pred(G1, h).cuda()
            loss = criterion1(pred1[train_mask1], edge_label1[train_mask1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if epoch % 10 == 0:
            #     print('Training acc:', compute_accuracy(pred1[train_mask1], edge_label1[train_mask1]), loss)

        h = self.gnn(G1, nfeats, efeats).cuda()
        pred1 = self.pred(G1, h).cuda()
        return pred1, edge_label1
    
    def predict(self, data1):
        G1_test = self.preprocessing.test(data1)
        G1_test = G1_test.to('cuda:0')
        actual1 = G1_test.edata.pop('label')
        node_features_test1 = G1_test.ndata['feature']
        edge_features_test1 = G1_test.edata['h']
        h = self.gnn(G1_test, node_features_test1, edge_features_test1)
        pred2 = self.pred(G1_test, h)
        return pred2, actual1
    
    # def forward(self, g, nfeats, efeats):
        # h = self.gnn(g, nfeats, efeats)
        # # h = list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
        # return self.pred(g, h)

# -------------------------------------------------------------------------------------------------------------------------------




# Loading data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
xai_datafile = "./input/Dataset/XAI/XAI_Test.csv"
gen_xai_testset = pd.read_csv(xai_datafile, encoding="ISO-8859–1", dtype = str)

# Label column is str dtype so we convert it to numpy.int64 dtype
gen_xai_testset["label"] = gen_xai_testset["label"].apply(lambda x: int(x))

# # Same thing with h attr, need to be converted to a list
# for index, row in gen_xai_testset.iterrows():
#     # print(row['h'])
#     # Remove brackets from the str
#     row['h'] = row['h'].replace("[", "")
#     row['h'] = row['h'].replace("]", "")
#     # Split depending on the seperator to have a list of str
#     row['h'] = row['h'].split(',')
#     # Convert str to float
#     row['h'] = [float(i) for i in row['h']]
#     gen_xai_testset.at[index,'h'] = row['h']
#     # print(type(row['h'][0]))

# labels_column = gen_xai_testset.label

print(gen_xai_testset["label"].value_counts())
print(gen_xai_testset)

# # Create our Multigraph
# XAI_G1 = nx.from_pandas_edgelist(gen_xai_testset, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
# XAI_G1 = XAI_G1.to_directed()
# XAI_G1 = from_networkx(XAI_G1, edge_attrs=['h','label'] )

# XAI_G1.ndata['h'] = th.ones(XAI_G1.num_nodes(), XAI_G1.edata['h'].shape[1])
# XAI_G1.edata['train_mask'] = th.ones(len(XAI_G1.edata['h']), dtype=th.bool)
# # Reshape both tensor lists to a single value in each element for both axis
# XAI_G1.ndata['h'] = th.reshape(XAI_G1.ndata['h'], (XAI_G1.ndata['h'].shape[0], 1, XAI_G1.ndata['h'].shape[1]))
# XAI_G1.edata['h'] = th.reshape(XAI_G1.edata['h'], (XAI_G1.edata['h'].shape[0], 1, XAI_G1.edata['h'].shape[1]))

# XAI_G1 = XAI_G1.to('cuda:0')

# node_features1 = XAI_G1.ndata['h']
# edge_features1 = XAI_G1.edata['h']
# edge_label1 = XAI_G1.edata['label']
# train_mask1 = XAI_G1.edata['train_mask']

# print(XAI_G1)
# print(edge_features1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# Loading the model ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nbclasses =  2

# Model *******************************************************************************************
# G1.ndata['h'].shape[2] = sizeh = 76 dans ANIDS
# model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
model1 = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
opt = th.optim.Adam(model1.parameters())
model1.load_state_dict(th.load("./models/Model1/model1_pre.pt"))
# model1.eval()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# Explain The model +++++++++++++++++++++++++++

print("start XAI")
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(model1.predict)
explainer = shap.KernelExplainer(model1.predict, gen_xai_testset)
shap_values = explainer.shap_values(gen_xai_testset)
# shap_values = explainer(edge_features1)
# shap_values = explainer(gen_xai_testset)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0], show = False)
plt.savefig('./notes/SHAP/grafic.png')

# +++++++++++++++++++++++++++++++++++++++++++++