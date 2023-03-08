import pandas as pd
import networkx as nx
import torch.nn as nn
from dgl import from_networkx
import torch as th
import matplotlib.pyplot as plt

import numpy as np

# dataframe
sizeh = 3
nbclasses =  2

columns=[" Source IP", " Destination IP", 'h','label']
data = [["192.168.10.25:222",2,[1,2,3],0], ["192.168.10.25:222",3,[1,20,3],1],["192.168.10.25:222",3,[2,2,3],0],["192.168.10.25:333",4,[3,2,3],0],[1,2,[1,2,4],0]]
X1_train = pd.DataFrame(data,columns=columns)

G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())
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
# plt.show()

## directed
G1 = G1.to_directed()

print()
print('******')
print("G1 : ", G1)
print("G1.nodes() : ", G1.nodes())
print("G1.nodes[1] : ", G1.nodes[1])
print("G1.edges(keys=True) : ", G1.edges(keys=True))
print("G1.edges.data() : ", G1.edges.data())
print('******')
print()



print('******')
att  = {}
# G1.nodes() :  [1, 2, 3, 4]
for n in G1.nodes():
    x = np.ones(sizeh)
    att[n] = {'h':x, 'nb':1}          
nx.set_node_attributes(G1, att)

att  = {}
# G1.edges(keys=True) :  [(1, 2, 0), (1, 2, 1), (1, 3, 0), (2, 1, 0), (2, 1, 1), (2, 3, 0), (3, 2, 0), (3, 1, 0), (3, 4, 0), (4, 3, 0)]
for n in G1.edges(keys=True): 
    x = np.ones(sizeh)
    att[n] = {'g':x}
nx.set_edge_attributes(G1, att)


print("G1 : ", G1)
print("G1.nodes() : ", G1.nodes())
print("G1.nodes[1] : ", G1.nodes[1])
print("G1.edges(keys=True) : ", G1.edges(keys=True))
print("G1.edges.data() : ", G1.edges.data())
print('******')
print()







print("##################################")
for node1, node2, data in G1.edges(data=True):
    G1.nodes[node2]['h'] = G1.nodes[node2]['h'] + data['h']
    G1.nodes[node2]['nb'] = G1.nodes[node2]['nb'] + 1
    G1.nodes[node2]['g'] = np.ones(sizeh)

for node2 in G1.nodes:
    G1.nodes[node2]['h'] = G1.nodes[node2]['h'] / G1.nodes[node2]['nb'] 

print("G1.nodes[1] : ", G1.nodes[1])
print("G1.edges.data() : ", G1.edges.data())
print("##################################")


print(G1.nodes())

print(mmmmmmmmmm)

G1 = from_networkx(G1,edge_attrs=['h','label'] )
# nodes data // G1.edata['h'].shape[1] : sizeh = number of attributes in a packet
G1.ndata['h'] = th.ones(G1.num_nodes(), G1.edata['h'].shape[1])
# edges data // we create a tensor bool array that will represent the train mask
G1.edata['train_mask'] = th.ones(len(G1.edata['h']), dtype=th.bool)

# Reshape both tensor lists to a single value in each element for both axis
G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
G1.edata['h'] = th.reshape(G1.edata['h'], (G1.edata['h'].shape[0], 1, G1.edata['h'].shape[1]))

# print(G1)
# print(G1.ndata)
# print(G1.edata)
# print(G1.edata['h'].shape[1])


## use of model
from sklearn.utils import class_weight
class_weights1 = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(G1.edata['label'].cpu().numpy()),
                                                  y = G1.edata['label'].cpu().numpy())
class_weights1 = th.FloatTensor(class_weights1).cuda()
criterion1 = nn.CrossEntropyLoss(weight=class_weights1)

G1 = G1.to('cuda:0')
# print(G1.device)
# print(G1.ndata['h'].device)
# print(G1.edata['h'].device)

print(G1)
# print(G1.ndata['h'].shape[0])
# print(G1.ndata['h'].shape[1])
# print(G1.ndata['h'].shape[2])
print()
print(G1.nodes())
print(G1.edges())

print()
print(G1.ndata)
print(G1.edata)