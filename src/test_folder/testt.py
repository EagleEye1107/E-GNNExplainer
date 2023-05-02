import dgl
import torch
from dgl import LineGraph
import pandas as pd
import networkx as nx
import torch as th
from dgl import from_networkx


columns=[" Source IP", " Destination IP", 'h','label']
data = [[1,2,[1,2,3],0], [2,3,[1,20,3],1],[1,3,[2,2,3],0],[3,4,[3,2,3],0],[1,2,[1,2,4],0]]
X1_train = pd.DataFrame(data,columns=columns)


G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiDiGraph())
print("initial nx multigraph G1 : ", G1)

# Convert it to a directed Graph
# NB : IT WILL CREATE A DEFAULT BIDIRECTIONAL RELATIONSHIPS BETWEEN NODES, and not the original relationships ???????????????????????
# G1 = G1.to_directed()
# print("G1 after todirected : ", G1)
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


print(G1.ndata)
print(G1.edata)
print("+++++")

transform = LineGraph()
G2 = transform(G1)

print(G2.ndata)
print(G2.edata)