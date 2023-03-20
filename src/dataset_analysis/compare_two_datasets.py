import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler


# # CIC-IDS-2017 *************************************************
# data1 = pd.read_csv('./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset0.csv', encoding="ISO-8859–1", dtype = str)

# # Delete two columns (U and V in the excel)
# cols = list(set(list(data1.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
# data1 = data1[cols]
# # Mise en forme des noeuds
# data1[' Source IP'] = data1[' Source IP'].apply(str)
# data1[' Source Port'] = data1[' Source Port'].apply(str)
# data1[' Destination IP'] = data1[' Destination IP'].apply(str)
# data1[' Destination Port'] = data1[' Destination Port'].apply(str)
# data1[' Source IP'] = data1[' Source IP'] + ':' + data1[' Source Port']
# data1[' Destination IP'] = data1[' Destination IP'] + ':' + data1[' Destination Port']
# data1.drop(columns=['Flow ID',' Source Port',' Destination Port',' Timestamp'], inplace=True)
# # -------------------- ????????????????????????????????????????? --------------------
# # simply do : nom = list(data1[' Label'].unique())
# nom = []
# nom = nom + [data1[' Label'].unique()[0]]
# for i in range(1, len(data1[' Label'].unique())):
#     nom = nom + [data1[' Label'].unique()[i]]
# nom.insert(0, nom.pop(nom.index('BENIGN')))
# # Naming the two classes BENIGN {0} / Any Intrusion {1}
# data1[' Label'].replace(nom[0], 0,inplace = True)
# for i in range(1,len(data1[' Label'].unique())):
#     data1[' Label'].replace(nom[i], 1,inplace = True)
# data1.rename(columns={" Label": "label"},inplace = True)
# label1 = data1.label
# data1.drop(columns=['label'],inplace = True)
# # split train and test
# data1 =  pd.concat([data1, label1], axis=1) # ??????? WHY ?

# print("nb Train instances : ", len(data1.values))
# # X_test = pd.concat([X_test, X1_test], ignore_index = True)

# # for non numerical attributes (categorical data)
# # Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
# # TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
# encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
# encoder1.fit(data1, label1)
# data1 = encoder1.transform(data1)

# # scaler (normalization)
# scaler1 = StandardScaler()

# # Manipulate flow content (all columns except : label, Source IP & Destination IP)
# cols_to_norm1 = list(set(list(data1.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
# data1[cols_to_norm1] = scaler1.fit_transform(data1[cols_to_norm1])

# ## Create the h attribute that will contain the content of our flows
# data1['h'] = data1[ cols_to_norm1 ].values.tolist()
# # print(data1)

# # size of the list containig the content of our flows
# sizeh1 = len(cols_to_norm1)


# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # Before training the data :
# # We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
# data1.drop(columns = cols_to_norm1, inplace = True)

# # Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
# columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
# data1=data1.reindex(columns=columns_titles)
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# print("sizeh = ", sizeh1)







# CIC-IDS-2018 *************************************************
data2 = pd.read_csv('./input/Dataset/CICIDS2018/02-14-2018.csv', encoding="ISO-8859–1", dtype = str)

# Create Src and Dst IP:Port columns in a stochastic way **************************************************
# From the xps we did on the CIC-IDS-2017, we noticed that we had 122661 Different IP:Port in the test set, lets use this info to create them

src_column = list(range(len(data2.values)))
dst_column = list(range(len(data2.values), 2 * len(data2.values)))

data2.insert(loc=0, column='Destination IP', value = dst_column)
data2.insert(loc=0, column='Source IP', value = src_column)

# Delete unnecessary columns (U and V in the excel)
cols = list(set(list(data2.columns )) - set(list(['Flow Byts/s','Flow Pkts/s', 'Dst Port', 'Timestamp'])) )
data2 = data2[cols]

# # Mise en forme des noeuds
data2['Source IP'] = data2['Source IP'].apply(str)
data2['Destination IP'] = data2['Destination IP'].apply(str)

# -------------------- ????????????????????????????????????????? --------------------
# simply do : nom = list(data2[' Label'].unique())
nom = []
nom = nom + [data2['Label'].unique()[0]]
for i in range(1, len(data2['Label'].unique())):
    nom = nom + [data2['Label'].unique()[i]]

nom.insert(0, nom.pop(nom.index('Benign')))

# Naming the two classes Benign {0} / Any Intrusion {1}
data2['Label'].replace(nom[0], 0,inplace = True)
for i in range(1,len(data2['Label'].unique())):
    data2['Label'].replace(nom[i], 1,inplace = True)

data2.rename(columns={"Label": "label"},inplace = True)
label2 = data2.label
data2.drop(columns=['label'],inplace = True)
# split train and test
data1 =  pd.concat([data2, label2], axis=1) # ??????? WHY ?


encoder2 = ce.TargetEncoder(cols=['Protocol',  'Fwd PSH Flags', 'Fwd URG Flags', 'Bwd PSH Flags', 'Bwd URG Flags'])
encoder2.fit(data2, label2)
data2 = encoder2.transform(data2)

# scaler (normalization)
scaler2 = StandardScaler()

# Manipulate flow content (all columns except : label, Source IP & Destination IP)
cols_to_norm2 = list(set(list(data2.iloc[:, :].columns )) - set(list(['label', 'Source IP', 'Destination IP'])) )
data2[cols_to_norm2] = scaler2.fit_transform(data1[cols_to_norm2])

## Create the h attribute that will contain the content of our flows
data2['h'] = data2[ cols_to_norm2 ].values.tolist()
# print(data1)

# size of the list containig the content of our flows
sizeh2 = len(cols_to_norm2)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Before training the data :
# We need to delete all the attributes (cols_to_norm1) to have the {Source IP, Destination IP, label, h} representation
data2.drop(columns = cols_to_norm2, inplace = True)

# Then we need to Swap {label, h} Columns to have the {Source IP, Destination IP, h, label} representation
columns_titles = ['Source IP', 'Destination IP', 'h', 'label']
data2 = data2.reindex(columns=columns_titles)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


print(cols_to_norm2)
print("sizeh2 = ", sizeh2)




print(data2)