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

#constante
size_embedding = 152
nb_batch = 1

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
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
            # Save edge_embeddings
            # nf = 'edge_embeddings'+str(i)+'.txt'
            # sourceFile = open(nf, 'w')
            # print(nfeats, file = sourceFile)
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
        # if(pr == True):
            # sourceFile = open(filename, 'w')
            # if pr:
                # print(v, file = sourceFile)
            # sourceFile.close()
        score = self.W(v)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            # Update the features of the specified edges by the provided function
            # DGLGraph.apply_edges(func, edges='__ALL__', etype=None, inplace=False)
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, nbclasses)
    def forward(self, g, nfeats, efeats, eweight = None):
        if eweight != None:
            # apply eweight on the graph
            efe = []
            for i, x in enumerate(eweight):
                efe.append(list(th.Tensor.cpu(g.edata['h'][i][0]).detach().numpy() * th.Tensor.cpu(x).detach().numpy()))

            efe = th.FloatTensor(efe).cuda()
            efe = th.reshape(efe, (efe.shape[0], 1, efe.shape[1]))
            g.edata['h'] = efe = efe

        h = self.gnn(g, nfeats, efeats)
        # h = list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
        return self.pred(g, h)

# -------------------------------------------------------------------------------------------------------------------------------



# # --------------------------------------------------- MAIN -----------------------------------------------------------

#Data
nbclasses =  2


# Model *******************************************************************************************
# G1.ndata['h'].shape[2] = sizeh = 76 dans ANIDS
# model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
model1 = Model(76, size_embedding, 76, F.relu, 0.2).cuda()
opt = th.optim.Adam(model1.parameters())



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

    # Create mini batches on the Train set
    # 1st step : Duplicate instances of least populated classes (nb occ < 100 => x100)
    for indx, x in enumerate(X1_train["label"].value_counts()) :
        if x < 100 :
            inst = X1_train.loc[X1_train['label'] == X1_train["label"].value_counts().index[indx]]
            for i in range(int(100 / x)) :
                X1_train = pd.concat([X1_train, inst], ignore_index = True)
    
    X1_train = shuffle(X1_train)
    
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

        # Each batch will contain 64500 instance and all classes are present (The least populated one has > 10 instances)

        print("nb Train instances : ", len(X1_train_batched.values))

        # for non numerical attributes (categorical data)
        # Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
        # TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
        encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
        encoder1.fit(X1_train_batched, y1_train_batched)
        X1_train_batched = encoder1.transform(X1_train_batched)

        # scaler (normalization)
        scaler1 = StandardScaler()

        # Manipulate flow content (all columns except : label, Source IP & Destination IP)
        cols_to_norm1 = list(set(list(X1_train_batched.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
        X1_train_batched[cols_to_norm1] = scaler1.fit_transform(X1_train_batched[cols_to_norm1])

        ## Create the h attribute that will contain the content of our flows
        X1_train_batched['h'] = X1_train_batched[ cols_to_norm1 ].values.tolist()
        # size of the list containig the content of our flows
        sizeh = len(cols_to_norm1)


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Before training the data :
        # We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
        X1_train_batched.drop(columns = cols_to_norm1, inplace = True)

        # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
        columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
        X1_train_batched = X1_train_batched.reindex(columns=columns_titles)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # ------------------------------------------- Creating the Graph Representation -------------------------------------------------------------
        # Create our Multigraph
        G1 = nx.from_pandas_edgelist(X1_train_batched, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiDiGraph())
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
        # ------------------------------------------- --------------------------------- -------------------------------------------------------------

        # ------------------------------------------- Model -----------------------------------------------------------------------------------------
        ## use of model
        from sklearn.utils import class_weight
        class_weights1 = class_weight.compute_class_weight(class_weight = 'balanced',
                                                        classes = np.unique(G1.edata['label'].cpu().numpy()),
                                                        y = G1.edata['label'].cpu().numpy())
        ''' 
            Using class weights, you make the classifier aware of how to treat the various classes in the loss function.
            In this process, you give higher weights to certain classes & lower weights to other classes.
            Example : [ 0.51600999 16.11525117] 
            Basically : 
                - For classes with small number of training images, you give it more weight
                so that the network will be punished more if it makes mistakes predicting the label of these classes. 
                - For classes with large numbers of images, you give it small weight
        '''
        class_weights1 = th.FloatTensor(class_weights1).cuda()
        criterion1 = nn.CrossEntropyLoss(weight = class_weights1)
        G1 = G1.to('cuda:0')

        node_features1 = G1.ndata['h']
        edge_features1 = G1.edata['h']

        edge_label1 = G1.edata['label']
        train_mask1 = G1.edata['train_mask']


        # to print
        pr = True
        # True if you want to print the embedding vectors
        # the name of the file where the vectors are printed
        filename = './models/M1_weights.txt'

        for epoch in range(1,1):
            pred = model1(G1, node_features1, edge_features1).cuda()
            loss = criterion1(pred[train_mask1], edge_label1[train_mask1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print('Training acc:', compute_accuracy(pred[train_mask1], edge_label1[train_mask1]), loss)

        pred1 = model1(G1, node_features1, edge_features1).cuda()
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
    X1_test = encoder1.transform(X1_test)
    X1_test[cols_to_norm1] = scaler1.transform(X1_test[cols_to_norm1])

    # Save X1_test for XAI
    X1_test.to_csv(f'./input/Dataset/XAI/X_test{nb_files}.csv', sep=',', index = False)

    X1_test['h'] = X1_test[ cols_to_norm1 ].values.tolist()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Before training the data :
    # We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
    X1_test.drop(columns = cols_to_norm1, inplace = True)

    # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
    columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
    X1_test=X1_test.reindex(columns=columns_titles)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    G1_test = nx.from_pandas_edgelist(X1_test, " Source IP", " Destination IP", ['h','label'],create_using=nx.MultiDiGraph())
    # G1_test = G1_test.to_directed()
    G1_test = from_networkx(G1_test,edge_attrs=['h','label'] )
    actual1 = G1_test.edata.pop('label')
    G1_test.ndata['feature'] = th.ones(G1_test.num_nodes(), G1.ndata['h'].shape[2])
    G1_test.ndata['feature'] = th.reshape(G1_test.ndata['feature'], (G1_test.ndata['feature'].shape[0], 1, G1_test.ndata['feature'].shape[1]))
    G1_test.edata['h'] = th.reshape(G1_test.edata['h'], (G1_test.edata['h'].shape[0], 1, G1_test.edata['h'].shape[1]))
    G1_test = G1_test.to('cuda:0')
    node_features_test1 = G1_test.ndata['feature']
    edge_features_test1 = G1_test.edata['h']

    # to print
    pr = True
    # True if you want to print the embedding vectors
    # the name of the file where the vectors are printed
    filename = './models/M1_weights.txt'

    print("nb instances : ", len(X1_test.values))

    test_pred1 = model1(G1_test, node_features_test1, edge_features_test1).cuda()
    test_pred1 = test_pred1.argmax(1)
    test_pred1 = th.Tensor.cpu(test_pred1).detach().numpy()

    print('Metrics : ')
    print("Accuracy : ", sklearn.metrics.accuracy_score(actual1, test_pred1))
    print("Precision : ", sklearn.metrics.precision_score(actual1, test_pred1, labels = [0,1]))
    print("Recall : ", sklearn.metrics.recall_score(actual1, test_pred1, labels = [0,1]))
    print("f1_score : ", sklearn.metrics.f1_score(actual1, test_pred1, labels = [0,1]))

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# Save the last Test Graph for XAI after
# graph_labels = {"glabel": th.tensor([0, 1])}
# save_graphs("./notes/DGLGraphs/data.bin", [G1_test], actual1)

# Save the model
# th.save(model1.state_dict(), "./models/Final_Model/modelF.pt")





# Explanation ***********************************************************************
from math import sqrt
from tqdm import tqdm
from dgl import EID, NID, khop_out_subgraph



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



# init mask
def init_masks(graph, efeat):
    # efeat.size() = torch.Size([nb_edges, 1, 76])
    efeat_size = efeat.size()[2]
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()

    device = efeat.device

    std = 0.1
    # feat_mask = [[f1, f2, .... fn]] / n = nb_features
    efeat_mask = nn.Parameter(th.randn(1, efeat_size, device=device) * std)

    std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
    # edge_mask = [e1, e2, .... em] / m = nb_edges
    edge_mask = nn.Parameter(th.randn(num_edges, device=device) * std)

    # print("efeat_mask : ", efeat_mask)
    # print("edge_mask : ", edge_mask)

    return efeat_mask, edge_mask


# Regularization loss
def loss_regularize(loss, feat_mask, edge_mask):
    # epsilon for numerical stability
    eps = 1e-15
    # From self GNNExplainer self
    alpha1 = 0.005,
    alpha2 = 1.0
    beta1 = 1.0
    beta2 = 0.1

    edge_mask = edge_mask.sigmoid()
    # Edge mask sparsity regularization
    loss = loss + th.from_numpy(alpha1 * th.Tensor.cpu(th.sum(edge_mask)).detach().numpy()).cuda()
    # Edge mask entropy regularization
    ent = -edge_mask * th.log(edge_mask + eps) - (
        1 - edge_mask
    ) * th.log(1 - edge_mask + eps)
    loss = loss + alpha2 * ent.mean()

    feat_mask = feat_mask.sigmoid()
    # Feature mask sparsity regularization
    loss = loss + beta1 * th.mean(feat_mask)
    # Feature mask entropy regularization
    ent = -feat_mask * th.log(feat_mask + eps) - (
        1 - feat_mask
    ) * th.log(1 - feat_mask + eps)
    loss = loss + beta2 * ent.mean()

    return loss


def explain_edge(model, edge_id, graph, node_feat, edge_feat, **kwargs):
    model = model.to(graph.device)
    model.eval()
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    print("graph : ", graph)
    print("graph.nodes() : ", graph.nodes())
    print("graph.edges() : ", graph.edges())

    # Extract source node-centered k-hop subgraph from the edge_id and its associated node and edge features.
    num_hops = 3
    source_node = th.Tensor.cpu(graph.edges()[0][edge_id]).detach().numpy()
    print("edge_id : ", edge_id)
    print("source_node : ", source_node)
    sg, inverse_indices = khop_out_subgraph(graph, source_node, num_hops)

    print("inverse_indices : ", inverse_indices)

    # EID = NID = _ID
    # tensor([0, 1, 2, 4]) : nodes and edges ids
    sg_edges = sg.edata[EID].long()
    sg_nodes = sg.ndata[NID].long()

    print("+++++++++++++++++++++++")
    print("sg : ", sg)
    print("sg_edges : ", sg_edges) # edges ids in graph.edges()
    print("sg_nodes : ", sg_nodes) # nodes ids in graph.nodes()

    print("+++++++++++++++++++++++")
    edge_feat = edge_feat[sg_edges]
    node_feat = node_feat[sg_nodes]

    print(edge_feat)
    print(node_feat)

    # Everything id good for now

    # If we add kwargs
    # for key, item in kwargs.items():
    #     if th.is_tensor(item) and item.size(0) == num_nodes:
    #         item = item[sg_nodes]
    #     elif th.is_tensor(item) and item.size(0) == num_edges:
    #         item = item[sg_edges]
    #     kwargs[key] = item

    # Get the initial prediction.
    print("Get the initial prediction :")
    with th.no_grad():
        # logits = model(g = sg, nfeats = node_feat, efeats = edge_feat, **kwargs)
        logits = model(g = sg, nfeats = node_feat, efeats = edge_feat)
        pred_label = logits.argmax(dim=-1)
        # pred_label1 = logits.argmax(1)

    print("pred_label : ", pred_label)
    # print(pred_label1)

    #
    efeat_mask, edge_mask = init_masks(sg, edge_feat)

    params = [efeat_mask, edge_mask]
    # params = [efeat_mask]
    # lr=0.01
    optimizer = th.optim.Adam(params, lr = 0.01)

    # if self.log:
        # pbar = tqdm(total=self.num_epochs)
        # pbar.set_description(f"Explain node {node_id}")

    # num_epochs = 100
    print("***********************************")
    print(efeat_mask)
    print(edge_mask)
    print("***********************************")
    for _ in range(1000):
        optimizer.zero_grad()
        # Matrix multiplication
        h = edge_feat * efeat_mask.sigmoid()
        logits = model(g = sg, nfeats = node_feat, efeats = h, eweight=edge_mask.sigmoid())
        # logits = model(g = sg, nfeats = node_feat, efeats = h)
        log_probs = logits.log_softmax(dim=-1)
        loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
        loss = loss_regularize(loss, efeat_mask, edge_mask)
        loss.backward()
        optimizer.step()

        # log = True
        # if self.log:
            # pbar.update(1)

    # log = True
    # if self.log:
        # pbar.close()

    print("final results before sigmoid : ")
    print("efeat_mask : ", efeat_mask)
    print("edge_mask : ", edge_mask)
    print("***********************************")

    efeat_mask = efeat_mask.detach().sigmoid().squeeze()
    edge_mask = edge_mask.detach().sigmoid()

    return inverse_indices, sg, efeat_mask, edge_mask



inv_indices, sub_graph, efeat_mask, edge_mask = explain_edge(model1, 2, G1_test, node_features_test1, edge_features_test1)

print("final results : ")
print("efeat_mask : ", efeat_mask)
print("edge_mask : ", edge_mask)