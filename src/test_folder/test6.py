import csv
# import dgl.nn as dglnn
from dgl import from_networkx
from psutil import cpu_times
import sklearn
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
# import socket
# import struct
import random
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import os




data1 = pd.read_csv("./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset0.csv", encoding="ISO-8859–1", dtype = str)
data2 = data1[data1[' Label'].str.contains("BENIGN")].head(10)
data1 = data1[data1[' Source IP'].str.contains("172.16.0.1")].tail(10)

data1 = pd.concat([data1, data2], ignore_index=True)

print(data1[[' Source IP', ' Destination IP', ' Label']])
print(len(data1))

print("nb total instances in the file : ", len(data1.values))

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

data1.rename(columns={" Label": "label"},inplace = True)
label1 = data1.label
data1.drop(columns=['label'],inplace = True)

# ******** At this step data1 contains only the data without label column
# ******** The label column is stored in the label variale 

# split train and test
data1 =  pd.concat([data1, label1], axis=1) # ??????? WHY ?

# Is Graph Representation Important ?? *************************************************************************
print("data IP Addr before changing them : ")
print(data1[[' Source IP', ' Destination IP']])

dff = pd.DataFrame({'col1': list(range(len(data1.values))), 'col2': list(range(len(data1.values), 2 * len(data1.values)))})

data1[' Source IP'] = dff['col1']
data1[' Destination IP'] = dff['col2']

print()
print("data IP Addr after changing them : ")
print(data1[[' Source IP', ' Destination IP']])
# ***********************************************************************************

# -------------------- ????????????????????????????????????????? --------------------
# X will contain the label column due to the concatination made earlier !!
X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123, stratify= label1)


print("nb Train instances : ", len(X1_train.values))
# X_test = pd.concat([X_test, X1_test], ignore_index = True)

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
# print(X1_train)

# size of the list containig the content of our flows
sizeh = len(cols_to_norm1)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Before training the data :
# We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
X1_train.drop(columns = cols_to_norm1, inplace = True)

# Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
X1_train=X1_train.reindex(columns=columns_titles)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


X1_train = X1_train.head(20)




# ------------------------------------------- Testing with a simple example -----------------------------------------------------------------
# sizeh = 3
# nbclasses =  2

# columns=[" Source IP", " Destination IP", 'h','label']
# data = [[1,2,[1,2,3],0], [2,3,[1,20,3],1],[1,3,[2,2,3],0],[3,4,[3,2,3],0],[1,2,[1,2,4],0]]
# X1_train = pd.DataFrame(data, columns=columns)
# ------------------------------------------- ----------------------------- -----------------------------------------------------------------


# ------------------------------------------- Creating the Graph Representation -------------------------------------------------------------
# Create our Multigraph
G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())

nx.draw_planar(G1, with_labels = True)
plt.show()

print("initial nx multigraph G1 : ", G1)

print(G1.nodes())

# Convert it to a directed Graph
# NB : IT WILL CREATE A DEFAULT BIDIRECTIONAL RELATIONSHIPS BETWEEN NODES, and not the original relationships ???????????????????????
G1 = G1.to_directed()
print("G1 after todirected : ", G1)
# Convert the graph from a networkx Graph to a DGL Graph

list1 = list(reversed(range(28)))
list2 = [str(x) for x in list1]
npa = np.asarray(list2, dtype=str)

print(npa)


G1 = from_networkx(G1, node_attrs = ['10', '26', '25', '24', '23', '22', '21', '20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], edge_attrs=['h','label'] )


print(G1.nodes())