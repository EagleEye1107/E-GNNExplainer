''''
        This version of E-GraphSAGE works on ZeroDay Attacks Detection,
        The idea here is to train our model on some attack days (representing specific attacks, for example Monday and Tuesday) 
    and test it on other attack days (that will represent completely different attacks => ZeroDay Attacks, the other days)
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

from focal_loss.focal_loss import FocalLoss

# Confusion Matrix ------------------------------------------------------------
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
# -----------------------------------------------------------------------------

#constante
size_embedding = 16

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
            # Line 4 of algorithm 1
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
            #nf = 'weights'+str(i)+'.txt'
            #sourceFile = open(nf, 'w')
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
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
        if(pr == True):
            sourceFile = open(filename, 'w')
            if pr:
                print(v, file = sourceFile)
            sourceFile.close()
        score = self.W(v)
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
        self.pred = MLPPredictor(ndim_out, nbclasses)
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        # h = list of node features [[node1_feature1, node1_feature2, ...], [node2_feature1, node2_feature2, ...], ...]
        return self.pred(g, h)

# -------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------- MAIN -----------------------------------------------------------

#Data
nbclasses =  2


print("++++++++++++++++++++++++++++ Train ++++++++++++++++++++++++++++++++")
path, dirs, files = next(os.walk("./input/Dataset/ZeroDayAttacks_Split/Train/"))
file_count = len(files)

for nb_files in range(file_count):
    X1_train = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)

    print(f'{files[nb_files]} ++++++++++++++++++++++++++++++++++++++++++++++')
    print("nb total instances in the file : ", len(X1_train.values))
    
    # Delete two columns (U and V in the excel)
    cols = list(set(list(X1_train.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
    X1_train = X1_train[cols]

    # Mise en forme des noeuds
    X1_train[' Source IP'] = X1_train[' Source IP'].apply(str)
    X1_train[' Source Port'] = X1_train[' Source Port'].apply(str)
    X1_train[' Destination IP'] = X1_train[' Destination IP'].apply(str)
    X1_train[' Destination Port'] = X1_train[' Destination Port'].apply(str)
    X1_train[' Source IP'] = X1_train[' Source IP'] + ':' + X1_train[' Source Port']
    X1_train[' Destination IP'] = X1_train[' Destination IP'] + ':' + X1_train[' Destination Port']

    X1_train.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)

    # -------------------- ????????????????????????????????????????? --------------------
    # simply do : nom = list(X1_train[' Label'].unique())
    nom = []
    nom = nom + [X1_train[' Label'].unique()[0]]
    for i in range(1, len(X1_train[' Label'].unique())):
        nom = nom + [X1_train[' Label'].unique()[i]]
    
    nom.insert(0, nom.pop(nom.index('BENIGN')))

    # Naming the two classes BENIGN {0} / Any Intrusion {1}
    X1_train[' Label'].replace(nom[0], 0,inplace = True)
    for i in range(1,len(X1_train[' Label'].unique())):
        X1_train[' Label'].replace(nom[i], 1,inplace = True)
    
    ##################### LABELS FREQ #######################################
    print()
    print("labels freq after changing labels to binary")
    counts = list(X1_train[' Label'].value_counts().to_dict().items())
    for j, x in enumerate(counts):
        x = list(x)
        x[1] = x[1] / len(X1_train)
        counts[j] = x
    print({f'{files[nb_files]}' : counts})
    ##############################################################################

    X1_train.rename(columns={" Label": "label"},inplace = True)
    label1 = X1_train.label
    X1_train.drop(columns=['label'],inplace = True)

    # ******** At this step X1_train contains only the data without label column
    # ******** The label column is stored in the label variale 

    # split train and test
    X1_train =  pd.concat([X1_train, label1], axis=1) # ??????? WHY ?

    # -------------------- ????????????????????????????????????????? --------------------
    # X will contain the label column due to the concatination made earlier !!
    # X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123, stratify= label1)

    print("nb Train instances : ", len(X1_train.values))
    # X_test = pd.concat([X_test, X1_test], ignore_index = True)

    # for non numerical attributes (categorical data)
    # Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
    # TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
    encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
    encoder1.fit(X1_train, label1)
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

    # ------------------------------------------- Creating the Graph Representation -------------------------------------------------------------
    # Create our Multigraph
    G1 = nx.from_pandas_edgelist(X1_train, " Source IP", " Destination IP", ['h','label'], create_using=nx.MultiGraph())

    print("initial nx multigraph G1 : ", G1)

    # Convert it to a directed Graph
    # NB : IT WILL CREATE A DEFAULT BIDIRECTIONAL RELATIONSHIPS BETWEEN NODES, and not the original relationships ???????????????????????
    
    # Removing the bidirectional edges
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


    # ------------------------------------------- Model -----------------------------------------------------------------------------------------
    ## use of model
    from sklearn.utils import class_weight
    class_weights1 = class_weight.compute_class_weight(class_weight = 'balanced',
                                                    classes = np.unique(G1.edata['label'].cpu().numpy()),
                                                    y = G1.edata['label'].cpu().numpy())
    class_weights1 = th.FloatTensor(class_weights1).cuda()
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1)
    # criterion1 = FocalLoss(gamma = 5, weights = class_weights1)
    G1 = G1.to('cuda:0')

    node_features1 = G1.ndata['h']
    edge_features1 = G1.edata['h']

    edge_label1 = G1.edata['label']
    train_mask1 = G1.edata['train_mask']

    # to print
    pr = True
    # True if you want to print the embedding vectors
    # the name of the file where the vectors are printed
    filename = './models/M1_weights_ZeroDayAttacks.txt'

    # Model architecture
    # G1.ndata['h'].shape[2] = sizeh = 76 dans ANIDS
    model1 = Model(G1.ndata['h'].shape[2], size_embedding, G1.ndata['h'].shape[2], F.relu, 0.2).cuda()
    opt = th.optim.Adam(model1.parameters())

    # m = nn.Sigmoid()
    for epoch in range(1,1000):
        pred = model1(G1, node_features1, edge_features1).cuda()
        # loss = criterion1(m(pred[train_mask1]), edge_label1[train_mask1])
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

    print("edge_features1 : ", len(edge_features1))
    print("pred1 : ", len(pred1))
    print("edge_label1 : ", len(edge_label1))

    print('confusion matrix :')
    c = confusion_matrix(edge_label1, pred1)
    print(c)
    c[0][0]= c[0][0]/2
    c[1][0]= c[1][0]/2
    c[0][1]= c[0][1]/2
    c[1][1]= c[1][1]/2
    print(c)

    print('metrics :')
    print("Accuracy : ", sklearn.metrics.accuracy_score(edge_label1, pred1))
    print("Precision : ", sklearn.metrics.precision_score(edge_label1, pred1, labels=[0,1]))
    print("Recall : ", sklearn.metrics.recall_score(edge_label1, pred1, labels=[0,1]))
    print("f1_score : ", sklearn.metrics.f1_score(edge_label1, pred1, labels=[0,1]))
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")






# ------------------------------------------------ Test ---------------------------------------------------------------------
print("++++++++++++++++++++++++++++ Test ++++++++++++++++++++++++++++++++")
path, dirs, files = next(os.walk("./input/Dataset/ZeroDayAttacks_Split/Test/"))
file_count = len(files)

for nb_files in range(file_count):
    X1_test = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)

    print(f'{files[nb_files]} ++++++++++++++++++++++++++++++++++++++++++++++')
    print("nb total instances in the file : ", len(X1_test.values))
    
    # Delete two columns (U and V in the excel)
    cols = list(set(list(X1_test.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
    X1_test = X1_test[cols]

    # Mise en forme des noeuds
    X1_test[' Source IP'] = X1_test[' Source IP'].apply(str)
    X1_test[' Source Port'] = X1_test[' Source Port'].apply(str)
    X1_test[' Destination IP'] = X1_test[' Destination IP'].apply(str)
    X1_test[' Destination Port'] = X1_test[' Destination Port'].apply(str)
    X1_test[' Source IP'] = X1_test[' Source IP'] + ':' + X1_test[' Source Port']
    X1_test[' Destination IP'] = X1_test[' Destination IP'] + ':' + X1_test[' Destination Port']

    X1_test.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)

    # -------------------- ????????????????????????????????????????? --------------------
    # simply do : nom = list(X1_test[' Label'].unique())
    nom = []
    nom = nom + [X1_test[' Label'].unique()[0]]
    for i in range(1, len(X1_test[' Label'].unique())):
        nom = nom + [X1_test[' Label'].unique()[i]]
    
    nom.insert(0, nom.pop(nom.index('BENIGN')))

    # Naming the two classes BENIGN {0} / Any Intrusion {1}
    X1_test[' Label'].replace(nom[0], 0,inplace = True)
    for i in range(1,len(X1_test[' Label'].unique())):
        X1_test[' Label'].replace(nom[i], 1,inplace = True)
    
    ##################### LABELS FREQ #######################################
    print()
    print("labels freq after changing labels to binary")
    counts = list(X1_test[' Label'].value_counts().to_dict().items())
    for j, x in enumerate(counts):
        x = list(x)
        x[1] = x[1] / len(X1_test)
        counts[j] = x
    print({f'{files[nb_files]}' : counts})
    ##############################################################################

    X1_test.rename(columns={" Label": "label"},inplace = True)
    label1 = X1_test.label
    X1_test.drop(columns=['label'],inplace = True)

    # ******** At this step X1_test contains only the data without label column
    # ******** The label column is stored in the label variale 

    # split train and test
    X1_test =  pd.concat([X1_test, label1], axis=1) # ??????? WHY ?

    print("nb Test instances : ", len(X1_test.values))
    X1_test = encoder1.transform(X1_test)
    X1_test[cols_to_norm1] = scaler1.transform(X1_test[cols_to_norm1])
    X1_test['h'] = X1_test[ cols_to_norm1 ].values.tolist()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Before training the data :
    # We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
    X1_test.drop(columns = cols_to_norm1, inplace = True)

    # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
    columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
    X1_test=X1_test.reindex(columns=columns_titles)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    G1_test = nx.from_pandas_edgelist(X1_test, " Source IP", " Destination IP", ['h','label'],create_using=nx.MultiGraph())

    # Removing the bidirectional edges
    G1_test = G1_test.to_directed()

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
    filename = './models/M1_weights_ZeroDayAttacks.txt'

    print("nb instances : ", len(X1_test.values))

    test_pred1 = model1(G1_test, node_features_test1, edge_features_test1).cuda()


    test_pred1 = test_pred1.argmax(1)
    test_pred1 = th.Tensor.cpu(test_pred1).detach().numpy()

    # actual11 = ["Normal" if i == 0 else "Attack" for i in actual1]
    # test_pred11 = ["Normal" if i == 0 else "Attack" for i in test_pred1]

    print("Confusion matrix : ")
    c = confusion_matrix(actual1, test_pred1)
    print(c)
    c[0][0]= c[0][0]/2
    c[1][0]= c[1][0]/2
    c[0][1]= c[0][1]/2
    c[1][1]= c[1][1]/2
    print(c)

    print('Metrics : ')
    print("Accuracy : ", sklearn.metrics.accuracy_score(actual1, test_pred1))
    print("Precision : ", sklearn.metrics.precision_score(actual1, test_pred1, labels = [0,1]))
    print("Recall : ", sklearn.metrics.recall_score(actual1, test_pred1, labels = [0,1]))
    print("f1_score : ", sklearn.metrics.f1_score(actual1, test_pred1, labels = [0,1]))

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
