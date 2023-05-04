from math import sqrt
import torch
import torch as th
from torch import nn
from tqdm import tqdm
import dgl
import pandas as pd
import networkx as nx


# columns=[" Source IP", " Destination IP", 'h','label']
# data = [[0,1,[1,2,3],0], [1,2,[1,20,3],1], [0,2,[2,2,3],0], [2,3,[3,2,3],0], [1,4,[1,2,4],0], [4,5,[1,2,4],0]]
# X_trr = pd.DataFrame(data,columns=columns)

# G1_test = nx.from_pandas_edgelist(X_trr, " Source IP", " Destination IP", ['h','label'], create_using = nx.MultiDiGraph())
# # G1_test = G1_test.to_directed()
# G1_test = from_networkx(G1_test,edge_attrs=['h','label'] )
# actual1 = G1_test.edata.pop('label')
# G1_test.ndata['feature'] = th.ones(G1_test.num_nodes(), 76)
# G1_test.ndata['feature'] = th.reshape(G1_test.ndata['feature'], (G1_test.ndata['feature'].shape[0], 1, G1_test.ndata['feature'].shape[1]))
# G1_test.edata['h'] = th.reshape(G1_test.edata['h'], (G1_test.edata['h'].shape[0], 1, G1_test.edata['h'].shape[1]))
# # G1_test = G1_test.to('cuda:0')
# node_features_test1 = G1_test.ndata['feature']
# edge_features_test1 = G1_test.edata['h']



# dataframe
sizeh = 3
nbclasses =  2

edge_mask = th.randn(5).cuda()
print("edge_mask : ", edge_mask)

columns=[" Source IP", " Destination IP", 'h','label']
data = [[1,2,[1,2,3],0], [2,3,[1,20,3],1],[1,3,[2,2,3],0],[3,4,[3,2,3],0],[1,2,[1,2,4],0]]
X1_train = pd.DataFrame(data,columns=columns)

# Create our Multigraph
G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiDiGraph())
G1 = dgl.from_networkx(G1, edge_attrs=['h','label'] )
G1.ndata['h'] = th.ones(G1.num_nodes(), G1.edata['h'].shape[1])
G1.edata['train_mask'] = th.ones(len(G1.edata['h']), dtype=th.bool)
G1.ndata['h'] = th.reshape(G1.ndata['h'], (G1.ndata['h'].shape[0], 1, G1.ndata['h'].shape[1]))
G1.edata['h'] = th.reshape(G1.edata['h'], (G1.edata['h'].shape[0], 1, G1.edata['h'].shape[1]))

G1 = G1.to('cuda:0')

node_id = 0

print(G1.edges())
print(G1.edata)
print(dddddddddddddd)
print("********************")
print("edge_mask : ", edge_mask)
print(G1.edata['h'])

efe = []
for i, x in enumerate(edge_mask):
    efe.append(list(th.Tensor.cpu(G1.edata['h'][i][0]).detach().numpy() * th.Tensor.cpu(x).detach().numpy()))

efe = th.FloatTensor(efe).cuda()
efe = th.reshape(efe, (efe.shape[0], 1, efe.shape[1]))
G1.edata['h'] = efe
print(G1.edata['h'])

print(G1)

print(dddddd)



G1 = dgl.graph(([0, 0, 1, 1, 2, 4], [1, 2, 2, 4, 3, 5]))

print(G1.nodes())
print(G1.edges())

sg, inverse_indices = dgl.khop_out_subgraph(G1, 1, k=3)
print("++++++++++++++++++++++++")
print(sg.nodes())
print(sg.edges())

print(ddddd)


# init mask
def init_masks(self, graph, feat):
    num_nodes, feat_size = feat.size()
    num_edges = graph.num_edges()
    device = feat.device

    std = 0.1
    # feat_mask = [[f1, f2, .... fn]] / n = nb_features
    feat_mask = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

    std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
    # edge_mask = [e1, e2, .... em] / m = nb_edges
    edge_mask = nn.Parameter(torch.randn(num_edges, device=device) * std)

    return feat_mask, edge_mask


# Regularization loss
def loss_regularize(self, loss, feat_mask, edge_mask):
    # epsilon for numerical stability
    eps = 1e-15

    edge_mask = edge_mask.sigmoid()
    # Edge mask sparsity regularization
    loss = loss + self.alpha1 * torch.sum(edge_mask)
    # Edge mask entropy regularization
    ent = -edge_mask * torch.log(edge_mask + eps) - (
        1 - edge_mask
    ) * torch.log(1 - edge_mask + eps)
    loss = loss + self.alpha2 * ent.mean()

    feat_mask = feat_mask.sigmoid()
    # Feature mask sparsity regularization
    loss = loss + self.beta1 * torch.mean(feat_mask)
    # Feature mask entropy regularization
    ent = -feat_mask * torch.log(feat_mask + eps) - (
        1 - feat_mask
    ) * torch.log(1 - feat_mask + eps)
    loss = loss + self.beta2 * ent.mean()

    return loss


def explain_node(self, node_id, graph, feat, **kwargs):
    self.model = self.model.to(graph.device)
    self.model.eval()
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    # Extract node-centered k-hop subgraph and
    # its associated node and edge features.
    # num_hops = 1
    sg, inverse_indices = dgl.khop_in_subgraph(graph, node_id, self.num_hops)
    
    # EID = NID = _ID
    # tensor([0, 1, 2, 4]) : node ad edge ids
    sg_nodes = sg.ndata[dgl.NID].long()
    sg_edges = sg.edata[dgl.EID].long()
    
    feat = feat[sg_nodes]

    # If we add kwargs
    for key, item in kwargs.items():
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[sg_nodes]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[sg_edges]
        kwargs[key] = item

    # Get the initial prediction.
    with torch.no_grad():
        logits = self.model(graph=sg, feat=feat, **kwargs)
        pred_label = logits.argmax(dim=-1)

    # 
    feat_mask, edge_mask = init_masks(sg, feat)

    params = [feat_mask, edge_mask]
    # lr=0.01
    optimizer = torch.optim.Adam(params, lr=self.lr)

    if self.log:
        pbar = tqdm(total=self.num_epochs)
        pbar.set_description(f"Explain node {node_id}")

    # num_epochs = 100
    for _ in range(self.num_epochs):
        optimizer.zero_grad()
        # Matrix multiplication
        h = feat * feat_mask.sigmoid()
        logits = self.model(
            graph=sg, feat=h, eweight=edge_mask.sigmoid(), **kwargs
        )
        log_probs = logits.log_softmax(dim=-1)
        loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
        loss = loss_regularize(loss, feat_mask, edge_mask)
        loss.backward()
        optimizer.step()

        # log = True
        if self.log:
            pbar.update(1)

    # log = True
    if self.log:
        pbar.close()

    feat_mask = feat_mask.detach().sigmoid().squeeze()
    edge_mask = edge_mask.detach().sigmoid()

    return inverse_indices, sg, feat_mask, edge_mask