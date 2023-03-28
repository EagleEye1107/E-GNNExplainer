# Data Processing
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

# Modelling
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz

import os


# RF Model
rf = RandomForestClassifier()


# path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
# path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted_With_Monday/"))
file_count = len(files)

# Classes
clss = ['BENIGN', 'Web Attack Ã\x82Â\x96 XSS', 'Heartbleed', 'DoS Hulk', 'DDoS', 'PortScan', 'Web Attack Ã\x82Â\x96 Brute Force', 'FTP-Patator', 'Bot', 'DoS slowloris', 'DoS GoldenEye', 'DoS Slowhttptest', 'Web Attack Ã\x82Â\x96 Sql Injection', 'SSH-Patator', 'Infiltration']

# Classes mpping
clss_mpping = {}
cpt = 0
for x in clss:
    clss_mpping[x] = cpt
    cpt += 1

print(clss_mpping)

for nb_files in range(file_count):
    data1 = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)
    print(f'{files[nb_files]} ++++++++++++++++++++++++++++++++++++++++++++++')

    # Delete two columns (U and V in the excel)
    cols = list(set(list(data1.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
    data1 = data1[cols]

    # IF We want to use IP Adr and do the mapping of the ip addresses because the random forest doesn't take in consideration
    '''
    # Mise en forme des noeuds
    data1[' Source IP'] = data1[' Source IP'].apply(str)
    data1[' Source Port'] = data1[' Source Port'].apply(str)
    data1[' Destination IP'] = data1[' Destination IP'].apply(str)
    data1[' Destination Port'] = data1[' Destination Port'].apply(str)
    data1[' Source IP'] = data1[' Source IP'] + ':' + data1[' Source Port']
    data1[' Destination IP'] = data1[' Destination IP'] + ':' + data1[' Destination Port']

    data1.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)

    # IP Mapping *************************************************************************
    # We do tha mapping of test set only because its faster and it will generate totally new nodes from the train set
    test_res = set()
    for x in list(data1[' Source IP']) :
        test_res.add(x)
    for x in list(data1[' Destination IP']) :
        test_res.add(x)

    test_re = {}
    cpt = 0
    for x in test_res:
        test_re[x] = cpt
        cpt +=1

    print(data1)
    data1 = data1.replace({' Source IP': test_re})
    data1 = data1.replace({' Destination IP': test_re})
    print(data1)

    print()
    # ***********************************************************************************
    '''

    # Delete unnecessary str data
    data1.drop(columns=['Flow ID',' Source IP',' Destination IP',' Source Port',' Destination Port',' Timestamp'], inplace=True)

    # data1 = data1.fillna(0)

    # -------------------- ????????????????????????????????????????? --------------------
    # simply do : nom = list(data1[' Label'].unique())
    nom = []
    nom = nom + [data1[' Label'].unique()[0]]
    for i in range(1, len(data1[' Label'].unique())):
        nom = nom + [data1[' Label'].unique()[i]]

    nom.insert(0, nom.pop(nom.index('BENIGN')))

    # Classes mpping
    data1 = data1.replace({' Label': clss_mpping})

    data1.rename(columns={" Label": "label"},inplace = True)
    label1 = data1.label
    data1.drop(columns=['label'],inplace = True)

    # ******** At this step data1 contains only the data without label column
    # ******** The label column is stored in the label variale 

    # split train and test
    data1 =  pd.concat([data1, label1], axis=1) # ??????? WHY ?

    # Splitting the dataset to train and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123)


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

    # Random Forest Model Training
    print("Model Training")
    rf.fit(X1_train, y1_train)

    # Test *******************************************************
    X1_test = encoder1.transform(X1_test)
    X1_test[cols_to_norm1] = scaler1.transform(X1_test[cols_to_norm1])

    # Model Testing
    print("Model Testing")
    y_pred = rf.predict(X1_test)

    print('Metrics : ')
    print("Accuracy : ", sklearn.metrics.accuracy_score(y1_test, y_pred))
    print("weighted f1-score : ", sklearn.metrics.f1_score(y1_test, y_pred, labels = list(range(15)), average = 'weighted'))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")