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

from dgl.data.utils import save_graphs, load_graphs

# Load Data
data1 = pd.read_csv('./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset0.csv', encoding="ISO-8859–1", dtype = str)
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
print({'counts : ' : counts})
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

# Each batch will contain 64500 instance and all classes are present (The least populated one has > 10 instances)
print("nb Train instances : ", len(X1_train.values))

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

## Create the h attribute that will contain the content of our flows
X1_train['h'] = X1_train[ cols_to_norm1 ].values.tolist()
# size of the list containig the content of our flows
sizeh = len(cols_to_norm1)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Before training the data :
# We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
X1_train.drop(columns = cols_to_norm1, inplace = True)

# Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
X1_train = X1_train.reindex(columns=columns_titles)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ------------------------------------------- Creating the Graph Representation -------------------------------------------------------------
# Create our Multigraph
G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
print("initial nx multigraph G1 : ", G1)

# Convert it to a directed Graph
# NB : IT WILL CREATE A DEFAULT BIDIRECTIONAL RELATIONSHIPS BETWEEN NODES, and not the original relationships ???????????????????????
G1 = G1.to_directed()
print("G1 after todirected : ", G1)
# Convert the graph from a networkx Graph to a DGL Graph
G1 = from_networkx(G1,edge_attrs=['h','label'] )
print("G1.edata['h'] after converting it to a dgl graph : ", len(G1.edata['h']))

# nodes data // G1.edata['h'].shape[1] : sizeh = number of attributes in a flow
G1.ndata['h'] = th.ones(G1.num_nodes(), G1.edata['h'].shape[1])
# edges data // we create a tensor bool array that will represent the train mask
G1.edata['train_mask'] = th.ones(len(G1.edata['h']), dtype=th.bool)

# Reshape both tensor lists to a single value in each element for both axis
G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
G1.edata['h'] = th.reshape(G1.edata['h'], (G1.edata['h'].shape[0], 1, G1.edata['h'].shape[1]))
print("G1.edata['h'] after reshape : ", len(G1.edata['h']))
# ------------------------------------------- --------------------------------- -------------------------------------------------------------

# Save the last Test Graph for XAI after
print(G1)
save_graphs("./notes/DGLGraphs/data.bin", G1)


G1_Loaded = load_graphs("./notes/DGLGraphs/data.bin")
print(G1)