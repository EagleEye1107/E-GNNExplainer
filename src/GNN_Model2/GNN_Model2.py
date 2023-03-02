import csv
import dgl.nn as dglnn
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in , ndim_out) 
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        x =  th.cat([edges.src['h']]) 
        x.type(th.cuda.FloatTensor)
        y = self.W_msg(x)
        y = {'m': y}
        return y

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['g'] = efeats
            # Eq4
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            # Eq5          
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']

    
class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.c1 = self.layers.append(SAGELayer(ndim_in, edim, size_embedding, activation))
        self.c2 = self.layers.append(SAGELayer(size_embedding, edim, size_embedding, activation)) ##
        self.c3 = self.layers.append(SAGELayer(size_embedding, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, g, nfeats, efeats):
        g.ndata['s0'] = nfeats
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
            if(i==0):
                g.ndata['s1'] = nfeats
            if(i==1):
                g.ndata['s2'] = nfeats
            if(i==2):
                g.ndata['s3'] = nfeats
        return nfeats.sum(1)
    
    
    
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)
    

    def apply_edges(self, edges):
        h_u0 = edges.src['s0']
        h_u1 = edges.src['s1']
        h_u2 = edges.src['s2']
        h_u3 = edges.src['s3']
        
        h_v0 = edges.dst['s0']
        h_v1 = edges.dst['s1']
        h_v2 = edges.dst['s2']
        h_v3 = edges.dst['s3']
    
        v = th.cat([h_u0, h_u1, h_u2, h_u3, h_v0, h_v1, h_v2, h_v3], 2)
        #v = th.cat([h_u3,h_v3],2)
        #v = th.cat([edges.src['h'],edges.dst['h']],1)
        if(pr == True):
            sourceFile = open(filename, 'w')
            if pr:
                print(v, file = sourceFile)
            sourceFile.close()
            
        score = self.W(v)
        score = th.reshape(score, (score.shape[0], score.shape[2]))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        new_dim = ndim_out * 3 + ndim_in
        self.pred = MLPPredictor(new_dim, nbclasses)
        
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)






#constante
size_embedding = 20
# to print
pr = False
# True if you want to print the embedding vectors
# the name of the file where the vectors are printed
filename = './models/M2_weights.txt'



## small example dataset
# size of h vectors
sizeh = 3
# size of the embedding vectors
# nmber of classes
nbclasses =  2


# dataframe
columns=[" Source IP", " Destination IP", 'h','label']
data = [[1,2,[1,2,3],0], [2,3,[1,20,3],1],[1,3,[2,2,3],0],[3,4,[3,2,3],0],[1,2,[1,2,4],0]]
X1_train = pd.DataFrame(data,columns=columns)
data = [[1,2,[1,22,3],1], [2,4,[1,1,3],0],[1,3,[2,2,3],0],[2,4,[3,2,3],0],[1,4,[3,2,4],0]]
X1_test = pd.DataFrame(data,columns=columns)




# --------------------------------------------------- MAIN -----------------------------------------------------------
# ------------------------------------------ This part is similar to GNN1 --------------------------------------------
#Data
nbclasses =  2
data1 = pd.read_csv('./input/Dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

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

# labels and there count
print(data1[' Label'].value_counts())

# -------------------- ????????????????????????????????????????? --------------------
# simply do : nom = list(data1[' Label'].unique())
nom = []
nom = nom + [data1[' Label'].unique()[0]]
for i in range(1, len(data1[' Label'].unique())):
    nom = nom + [data1[' Label'].unique()[i]]

# Naming the two classes BENIGN {0} / Any Intrusion {1}
data1[' Label'].replace(nom[0], 0,inplace = True)
for i in range(1,len(data1[' Label'].unique())):
    data1[' Label'].replace(nom[i], 1,inplace = True)

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

# for non numerical attributes (categorical data)
# Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
# TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
encoder1.fit(X1_train, y1_train)
X1_train = encoder1.transform(X1_train)

# scaler (normalization)
scaler1 = StandardScaler()

# Manipulate flow content (all columns except : label, Source IP & Destination IP)
cols_to_norm1 = list(set(list(X1_train.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
X1_train[cols_to_norm1] = scaler1.fit_transform(X1_train[cols_to_norm1])

X1_train['h'] = X1_train[ cols_to_norm1 ].values.tolist()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Before training the data :
# We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
# X1_train.drop(columns = cols_to_norm1, inplace = True)

# # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
# columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
# X1_train=X1_train.reindex(columns=columns_titles)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


##test
X1_test = encoder1.transform(X1_test)
X1_test[cols_to_norm1] = scaler1.transform(X1_test[cols_to_norm1])
X1_test['h'] = X1_test[ cols_to_norm1 ].values.tolist()

sizeh = len(cols_to_norm1)


G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'],
                             create_using=nx.MultiDiGraph())
## directed
G1 = G1.to_directed()

# ------------------------------------------ This part is similar to GNN1 --------------------------------------------



#pretreatment to compute the vectors associated to the nodes: 
#the average values of the attributes associated to the edges for which the node is a destination
att  = {}
for n in G1.nodes():
    x = np.ones(sizeh)
    att[n] = {'h':x, 'nb':1}          
nx.set_node_attributes(G1, att)

att  = {}
for n in G1.edges(keys=True): 
    x = np.ones(sizeh)
    att[n] = {'g':x}
nx.set_edge_attributes(G1, att)

for node1, node2, data in G1.edges(data=True):
    G1.nodes[node2]['h'] = G1.nodes[node2]['h'] + data['h']
    G1.nodes[node2]['nb'] = G1.nodes[node2]['nb'] + 1
    G1.nodes[node2]['g'] = np.ones(sizeh)

for node2 in G1.nodes:
    G1.nodes[node2]['h'] = G1.nodes[node2]['h'] / G1.nodes[node2]['nb'] 

G1 = from_networkx(G1, edge_attrs=['g','label'], node_attrs=['h'] )

# create the variables to store the embedding vectors
G1.ndata['s0'] = th.ones(G1.num_nodes(), sizeh)
G1.ndata['s0'] = th.reshape(G1.ndata['s0'], (G1.ndata['s0'].shape[0], 1, G1.ndata['s0'].shape[1]))
G1.ndata['s1'] = th.ones(G1.num_nodes(), size_embedding)
G1.ndata['s1'] = th.reshape(G1.ndata['s1'], (G1.ndata['s1'].shape[0], 1, G1.ndata['s1'].shape[1]))
G1.ndata['s2'] = th.ones(G1.num_nodes(), size_embedding)
G1.ndata['s2'] = th.reshape(G1.ndata['s2'], (G1.ndata['s2'].shape[0], 1, G1.ndata['s2'].shape[1]))
G1.ndata['s3'] = th.ones(G1.num_nodes(), size_embedding)
G1.ndata['s3'] = th.reshape(G1.ndata['s3'], (G1.ndata['s3'].shape[0], 1, G1.ndata['s3'].shape[1]))

G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
G1.edata['g'] = th.reshape(G1.edata['g'], (G1.edata['g'].shape[0], 1, G1.edata['g'].shape[1]))
G1.ndata['h'] = th.tensor(G1.ndata['h'], dtype=th.float)
G1.edata['g'] = th.tensor(G1.edata['g'], dtype=th.float)



G1 = G1.to('cuda:0')
G1.device
G1.ndata['h'].device
G1.edata['g'].device
G1.ndata['h'].type(th.cuda.FloatTensor)

from sklearn.utils import class_weight
class_weights1 = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(G1.edata['label'].cpu().numpy()),
                                                  y = G1.edata['label'].cpu().numpy())
class_weights1 = th.FloatTensor(class_weights1).cuda()
criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='mean')



node_features1 = G1.ndata['h']
edge_features1 = G1.edata['g']
edge_label1 = G1.edata['label']


model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
opt = th.optim.Adam(model1.parameters())

for epoch in range(1,1000):
    pred = model1(G1, node_features1, edge_features1).cuda()
    loss = criterion1(pred, edge_label1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
      print('Training acc:', compute_accuracy(pred, edge_label1), loss)
print('Training acc:', compute_accuracy(pred, edge_label1), loss)
## embedding vectors
#print('v0', G1.ndata['s0'])
#print('v1', G1.ndata['s1'])
#print('v2', G1.ndata['s2'])
#print('v3', G1.ndata['s3'])

pred1 = model1(G1, node_features1, edge_features1).cuda()
pred1 = pred1.argmax(1)
pred1 = th.Tensor.cpu(pred1).detach().numpy()
edge_label1 = th.Tensor.cpu(edge_label1).detach().numpy()

cm = confusion_matrix(edge_label1, pred1)
print(cm)




'''
G1_test = nx.from_pandas_edgelist(X1_test, " Source IP", " Destination IP", ['h','label'],
                                  create_using=nx.MultiDiGraph())
G1_test = G1_test.to_directed()

#pretreatment to compute the vectors associated to the nodes: 
#the average values of the attributes associated to the edges for which the node is a destination
att  = {}
for n in G1_test.nodes():
    x = np.ones(sizeh)
    att[n] = {'h':x, 'nb':1}
                   
nx.set_node_attributes(G1_test, att)
att  = {}
for n in G1_test.edges(keys=True): 
    x = np.ones(sizeh)
    att[n] = {'g':x}
nx.set_edge_attributes(G1_test, att)
for node1, node2, data in G1_test.edges(data=True):
    G1_test.nodes[node2]['h'] = G1_test.nodes[node2]['h'] + data['h']
    G1_test.nodes[node2]['nb'] = G1_test.nodes[node2]['nb'] + 1
    G1_test.nodes[node2]['g'] = np.ones(sizeh)

for node2 in G1_test.nodes:
    G1_test.nodes[node2]['h'] = G1_test.nodes[node2]['h'] / G1_test.nodes[node2]['nb'] 
    
G1_test = from_networkx(G1_test, edge_attrs=['g','label'], node_attrs=['h'] )

# create the variables to store the embedding vectors
G1_test.ndata['s0'] = th.ones(G1_test.num_nodes(), sizeh)
G1_test.ndata['s0'] = th.reshape(G1_test.ndata['s0'], (G1_test.ndata['s0'].shape[0], 1, G1_test.ndata['s0'].shape[1]))
G1_test.ndata['s1'] = th.ones(G1_test.num_nodes(), size_embedding)
G1_test.ndata['s1'] = th.reshape(G1_test.ndata['s1'], (G1_test.ndata['s1'].shape[0], 1, G1_test.ndata['s1'].shape[1]))
G1_test.ndata['s2'] = th.ones(G1_test.num_nodes(), size_embedding)
G1_test.ndata['s2'] = th.reshape(G1_test.ndata['s2'], (G1_test.ndata['s2'].shape[0], 1, G1_test.ndata['s2'].shape[1]))
G1_test.ndata['s3'] = th.ones(G1_test.num_nodes(), size_embedding)
G1_test.ndata['s3'] = th.reshape(G1_test.ndata['s3'], (G1_test.ndata['s3'].shape[0], 1, G1_test.ndata['s3'].shape[1]))

G1_test.ndata['h'] = th.reshape(G1_test.ndata['h'], (G1_test.ndata['h'].shape[0], 1, G1_test.ndata['h'].shape[1]))
G1_test.edata['g'] = th.reshape(G1_test.edata['g'], (G1_test.edata['g'].shape[0], 1, G1_test.edata['g'].shape[1]))
G1_test.ndata['h'] = th.tensor(G1_test.ndata['h'], dtype=th.float)
G1_test.edata['g'] = th.tensor(G1_test.edata['g'], dtype=th.float)


G1_test = G1_test.to('cuda:0')
G1_test.device
G1_test.ndata['h'].device
G1_test.edata['g'].device
G1_test.ndata['h'].type(th.cuda.FloatTensor)

node_features_test1 = G1_test.ndata['h']
edge_features_test1 = G1_test.edata['g']
edge_label_test1 = G1_test.edata['label']
pred1 = model1(G1_test, node_features_test1, edge_features_test1).cuda()
pred1 = pred1.argmax(1)
pred1 = th.Tensor.cpu(pred1).detach().numpy()
edge_label_test1 = th.Tensor.cpu(edge_label_test1).detach().numpy()

cm = confusion_matrix(edge_label_test1, pred1)
print(cm)
'''