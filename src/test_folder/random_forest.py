# Data Processing
import pandas as pd
import numpy as np
import category_encoders as ce

# Modelling
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz





# data1 = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)
data1 = pd.read_csv('./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset0.csv', encoding="ISO-8859–1", dtype = str)

# Mise en forme des noeuds
data1[' Source IP'] = data1[' Source IP'].apply(str)
data1[' Source Port'] = data1[' Source Port'].apply(str)
data1[' Destination IP'] = data1[' Destination IP'].apply(str)
data1[' Destination Port'] = data1[' Destination Port'].apply(str)
data1[' Source IP'] = data1[' Source IP'] + ':' + data1[' Source Port']
data1[' Destination IP'] = data1[' Destination IP'] + ':' + data1[' Destination Port']

data1.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)

# We need to do the mapping of the ip addresses because the random forest doesn't take in consideration 
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

data1.rename(columns={" Label": "label"},inplace = True)
label1 = data1.label
data1.drop(columns=['label'],inplace = True)

# ******** At this step data1 contains only the data without label column
# ******** The label column is stored in the label variale 

# split train and test
data1 =  pd.concat([data1, label1], axis=1) # ??????? WHY ?




# Splitting the dataset to train and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123, stratify= label1)


# for non numerical attributes (categorical data)
# Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
# TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
encoder1.fit(X1_train, y1_train)
X1_train = encoder1.transform(X1_train)




rf = RandomForestClassifier()
rf.fit(X1_train, y1_train)




# Test *******************************************************
X1_test = encoder1.transform(X1_test)



y_pred = rf.predict(X1_test)



print('Metrics : ')
print("Accuracy : ", sklearn.metrics.accuracy_score(y1_test, y_pred))
print("Precision : ", sklearn.metrics.precision_score(y1_test, y_pred, labels = [0,1]))
print("Recall : ", sklearn.metrics.recall_score(y1_test, y_pred, labels = [0,1]))
print("f1_score : ", sklearn.metrics.f1_score(y1_test, y_pred, labels = [0,1]))