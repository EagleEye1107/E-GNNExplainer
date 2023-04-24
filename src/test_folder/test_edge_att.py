import networkx as nx
import pandas as pd
from dgl import from_networkx


columns=[" Source IP", " Destination IP", 'h','label']
data = [[1,2,[1,2,3],0], [2,3,[1,20,3],1],[1,3,[2,2,3],2],[3,4,[3,2,3],3],[1,2,[1,2,4],4]]
X1_train = pd.DataFrame(data,columns=columns)


G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'],create_using=nx.MultiDiGraph())





G1 = from_networkx(G1,edge_attrs=['h','label'] )
print(G1.srcdata)
print(G1.edata['label'])