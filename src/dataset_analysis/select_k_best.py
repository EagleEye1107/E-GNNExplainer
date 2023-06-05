import pandas as pd
import numpy as np
import os

path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
file_count = len(files)

data1 = pd.DataFrame()
for nb_files in range(file_count):
    datag = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)
    data1 = pd.concat([data1, datag], ignore_index = True)


print("nb total instances in the file : ", len(data1.values))
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

data1.rename(columns={" Label": "label"},inplace = True)
label1 = data1.label
data1.drop(columns=['label'],inplace = True)

# split train and test
# data1 =  pd.concat([data1, label1], axis=1)

cols = list(set(list(data1.columns )) - set(list([' Source IP', ' Destination IP'])) )
data1 = data1[cols]

##########
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
encoder1.fit(data1, label1)
data1 = encoder1.transform(data1)

scaler1 = StandardScaler()
cols_to_norm1 = list(set(list(data1.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
data1[cols_to_norm1] = scaler1.fit_transform(data1[cols_to_norm1])
##########


X = data1.values
Y = label1.values

print(Y)

print("********************")

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2


# Feature extraction
nb_features_to_select = 35
test = SelectKBest(score_func = mutual_info_classif, k = nb_features_to_select)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision = 3)
print(fit.scores_)

val = fit.scores_

feature_indx = []
for i in range(nb_features_to_select):
    f_indx = np.argmax(val)
    feature_indx.append(f_indx)
    val[f_indx] = float('-inf')

print(feature_indx)

print(data1.columns[feature_indx])
print(len(data1.columns[feature_indx]))

important_features = ['label', ' Source IP', ' Destination IP', 'Flow ID',' Source Port',' Destination Port',' Timestamp', 'Flow Bytes/s',' Flow Packets/s']
final_features = list(set(list(data1.columns[feature_indx]))) + list(set(list(important_features)))

print(final_features)
print(len(final_features))