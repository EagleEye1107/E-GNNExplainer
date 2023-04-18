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
nb_batch = 5

# --------------------------------------------------- MAIN -----------------------------------------------------------

#Data
nbclasses =  2

path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
file_count = len(files)

for nb_files in range(file_count):
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
        print(type(X1_train_batched.iloc[4]["label"]))
        print(type(X1_train_batched.iloc[4]["h"][0]))
