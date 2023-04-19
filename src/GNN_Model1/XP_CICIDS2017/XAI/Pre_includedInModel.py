''''
    This version of GNN1 is similar to the original,
    The only difference is that the test will be done after training on each dataset file
    So we will have 4 test phases (Train1 -> Test1 -> Train2 -> Test2 ...etc.)
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
from sklearn.model_selection import train_test_split
import category_encoders as ce
import numpy as np
from sklearn.metrics import confusion_matrix

import os
from sklearn.utils import shuffle

from dgl.data.utils import save_graphs

import shap
import matplotlib.pyplot as plt

#constante
size_embedding = 152
nb_batch = 1
# to print
pr = True
filename = './models/M1_weights.txt'

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
        self.encoder1 = ce.TargetEncoder(cols=['Fwd PSH Flags', ' Protocol', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
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



# --------------------------------------------------- MAIN -----------------------------------------------------------

#Data
nbclasses =  2


# Model *******************************************************************************************
# G1.ndata['h'].shape[2] = sizeh = 76 dans ANIDS
# model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
model1 = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
opt = th.optim.Adam(model1.parameters())


preprocessor1 = GPreprocessing()


path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
file_count = len(files)

for nb_files in range(1):
    data1 = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)

    print(f'{files[nb_files]} ++++++++++++++++++++++++++++++++++++++++++++++')
    print("nb total instances in the file : ", len(data1.values))

    print("++++++++++++++++++++++++++++ Train ++++++++++++++++++++++++++++++++")
    
    # Delete two columns (U and V in the excel)
    cols = list(set(list(data1.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
    data1 = data1[cols]

    # Mise en forme des noeuds
    data1[' Source IP'] = data1[' Source IP'].apply(str)
    data1[' Source Port'] = data1[' Source Port'].apply(str)
    data1[' Destination IP'] = data1[' Destination IP'].apply(str)
    data1[' Destination Port'] = data1[' Destination Port'].apply(str)
    data1[' Source IP'] = data1[' Source IP'] + ':' + data1[' Source Port']
    data1[' Destination IP'] = data1[' Destination IP'] + ':' + data1[' Destination Port']

    data1.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)

    # -------------------- ????????????????????????????????????????? --------------------
    # simply do : nom = list(data1[' Label'].unique())
    nom = []
    nom = nom + [data1[' Label'].unique()[0]]
    for i in range(1, len(data1[' Label'].unique())):
        nom = nom + [data1[' Label'].unique()[i]]
    
    nom.insert(0, nom.pop(nom.index('BENIGN')))

    # Naming the two classes BENIGN {0} / Any Intrusion {1}
    data1[' Label'].replace(nom[0], 0,inplace = True)
    for i in range(1,len(data1[' Label'].unique())):
        data1[' Label'].replace(nom[i], 1,inplace = True)
    
    ##################### LABELS FREQ #######################################
    print()
    print("labels freq after changing labels to binary")
    counts = list(data1[' Label'].value_counts().to_dict().items())
    for j, x in enumerate(counts):
        x = list(x)
        x[1] = x[1] / len(data1)
        counts[j] = x
    print({f'{files[nb_files]}' : counts})
    ##############################################################################

    data1.rename(columns={" Label": "label"},inplace = True)
    label1 = data1.label
    data1.drop(columns=['label'],inplace = True)

    # ******** At this step data1 contains only the data without label column
    # ******** The label column is stored in the label variale 

    # split train and test
    data1 =  pd.concat([data1, label1], axis=1) # ??????? WHY ?

    # -------------------- ????????????????????????????????????????? --------------------
    # X will contain the label column due to the concatination made earlier !!
    X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123, stratify= label1)

    
    # At this step we duplicated the least populated classes in the Train Set
    # 2nd step : Create the mini batches
    a = b = mean_macro_f1 = 0
    for batch in range(1, nb_batch + 1):
        print(f"+++++++++++++++++ Batch {batch} ++++++++++++++++")
        a = b
        b = int(len(X1_train) / nb_batch) * batch
        if batch == nb_batch :
            b = len(X1_train)
        # The batch :
        X1_train_batched = X1_train.iloc[a:b]
        # y1_train_batched = y1_train.iloc[a:b]
        y1_train_batched = X1_train_batched['label']

        pred1, edge_label1 = model1.train(X1_train_batched, 1)
        pred1 = pred1.argmax(1)
        pred1 = th.Tensor.cpu(pred1).detach().numpy()
        edge_label1 = th.Tensor.cpu(edge_label1).detach().numpy()

        print('Train metrics :')
        print("Accuracy : ", sklearn.metrics.accuracy_score(edge_label1, pred1))
        print("Precision : ", sklearn.metrics.precision_score(edge_label1, pred1, labels = [0,1]))
        print("Recall : ", sklearn.metrics.recall_score(edge_label1, pred1, labels = [0,1]))
        print("f1_score : ", sklearn.metrics.f1_score(edge_label1, pred1, labels=[0,1]))

    # ------------------------------------------------ Test ---------------------------------------------------------------------
    print("++++++++++++++++++++++++++++ Test ++++++++++++++++++++++++++++++++")
    print("nb Test instances : ", len(X1_test.values))
    
    test_pred1, actual1 = model1.predict(X1_test)
    
    test_pred1 = test_pred1.argmax(1)
    test_pred1 = th.Tensor.cpu(test_pred1).detach().numpy()
    actual1 = th.Tensor.cpu(actual1).detach().numpy()

    print('Metrics : ')
    print("Accuracy : ", sklearn.metrics.accuracy_score(actual1, test_pred1))
    print("Precision : ", sklearn.metrics.precision_score(actual1, test_pred1, labels = [0,1]))
    print("Recall : ", sklearn.metrics.recall_score(actual1, test_pred1, labels = [0,1]))
    print("f1_score : ", sklearn.metrics.f1_score(actual1, test_pred1, labels = [0,1]))

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# # Save the gen_xai_trainset (background dataset) gen_xai_testset which is the final global X1_test set
# X1_train_batched.to_csv(f'./input/Dataset/XAI/XAI_Train.csv', sep=',', index = False)
# X1_test.to_csv(f'./input/Dataset/XAI/XAI_Test.csv', sep=',', index = False)

# # Save the model
# th.save(model1.state_dict(), "./models/Model1/model1_pre.pt")

print(X1_test)
print(list(set(list(X1_test.columns))))

cols_to_norm1 = list(set(list(data1.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])))

X1_train_batched[cols_to_norm1] = X1_train_batched[cols_to_norm1].apply(pd.to_numeric)
X1_test[cols_to_norm1] = X1_test[cols_to_norm1].apply(pd.to_numeric)

print()
print(X1_test.dtypes)
print(X1_train_batched.dtypes)

# XAI ######################
explainer = shap.Explainer(model1.predict, X1_train_batched[cols_to_norm1])
shap_values = explainer(X1_test[cols_to_norm1])

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0], show = False)
plt.savefig('./notes/SHAP/grafic.png')
