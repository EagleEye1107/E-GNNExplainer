# import csv
# import dgl.nn as dglnn
from dgl import from_networkx, to_networkx
# import torch.nn as nn
import torch as th
# import torch.nn.functional as F
# import dgl.function as fn
import networkx as nx
import pandas as pd
# import socket
# import struct
# import random
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
# from sklearn.decomposition import PCA
# import seaborn as sns
import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix



data1 = pd.read_csv('./input/Dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')


#Data
nbclasses =  2

#data = dataset[0]
p = ''#data/cicids/TrafficLabelling/'
#data1 = pd.read_csv(p + 'Wednesday-workingHours.pcap_ISCX.csv')
#data1 = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
# data1 = pd.read_csv('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
###data1 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
#data1 = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv') -> benin
#data1 = pd.read_csv('Friday-WorkingHours-Morning.pcap_ISCX.csv')
#data1 = pd.read_csv('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')

# Delete two columns (U and V in the excel)
cols = list(set(list(data1.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
data1 = data1[cols]

##mise en forme des noeuds
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

#split train and co
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
# Manipulate Packet characteristics (all columns except : label, Source IP & Destination IP)
cols_to_norm1 = list(set(list(X1_train.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
X1_train[cols_to_norm1] = scaler1.fit_transform(X1_train[cols_to_norm1])


## Create the h attribute that will contain the characteristic of our packets
X1_train['h'] = X1_train[ cols_to_norm1 ].values.tolist()

X1_train.drop(columns = cols_to_norm1, inplace = True)

print(X1_train)

columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
X1_train=X1_train.reindex(columns=columns_titles)

print(X1_train)
print(len(X1_train['h'].values.tolist()[0]))



# ------------------------------------------- Creating the Graph Representation -------------------------------------------------------------
# Create our Multigraph
G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())

# Convert it to a directed Graph
# NB : IT WILL CREATE A DEFAULT BIDIRECTIONAL RELATIONSHIPS BETWEEN NODES, and not the original relationships ???????????????????????
G1 = G1.to_directed()
# Convert the graph from a networkx Graph to a DGL Graph
G1 = from_networkx(G1,edge_attrs=['h','label'] )

# nodes data // G1.edata['h'].shape[1] : sizeh = number of attributes in a packet
G1.ndata['h'] = th.ones(G1.num_nodes(), G1.edata['h'].shape[1])
# edges data // we create a tensor bool array that will represent the train mask
G1.edata['train_mask'] = th.ones(len(G1.edata['h']), dtype=th.bool)

# Reshape both tensor lists to a single value in each element for both axis
G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
G1.edata['h'] = th.reshape(G1.edata['h'], (G1.edata['h'].shape[0], 1, G1.edata['h'].shape[1]))
# ------------------------------------------- --------------------------------- -------------------------------------------------------------


G = to_networkx(G1)
nx.draw(G)
plt.show()





# G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
# print(G1.edges) # all edges and nodes and well stored in the graph

# # Visualize the directed graph
# posit = nx.spring_layout(G1)
# options = {
#     'node_color': 'blue',
#     'node_size': 500,
#     'width': 2,
#     'arrowstyle': '-|>',
#     'arrowsize': 6,
# }

# nx.draw_networkx(G1, arrows=True, pos=posit, **options)
# nx.draw_networkx_edge_labels(G1, pos=posit)
# plt.savefig("./notes/graphs/graph2.png")
# plt.show()