from math import sqrt
import torch
import torch as th
from torch import nn
from tqdm import tqdm
import dgl
import pandas as pd
import networkx as nx

# G1 = dgl.graph(([1, 1, 2, 3, 4], [0, 2, 0, 4, 2]))

# print(G1.nodes())
# print(G1.edges())
# sg, inverse_indices = dgl.khop_in_subgraph(G1, 1, k=1)
# print("++++++++++++++++++++++++")
# print(sg.nodes())
# print(sg.edges())

# print("++++++")
# print(sg.ndata[dgl.NID])
# print(sg.ndata[dgl.NID].long())

# print(ddddd)


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