import pandas as pd
from sklearn.model_selection import train_test_split


def func(data22):
    print("functionnnnnnnnnnnnn")
    data22[' Source IP'] = src_ip
    data22[' Destination IP'] = dst_ip
    data22['label'] = data_labels

    # samplll = data22.iloc[0:10]

    print(data22.iloc[0:10][[' Source IP', ' Destination IP', 'label']])


data1 = pd.read_csv('./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset4.csv', encoding="ISO-8859–1", dtype = str)
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
# print("labels freq after changing labels to binary")
counts = list(data1[' Label'].value_counts().to_dict().items())
for j, x in enumerate(counts):
    x = list(x)
    x[1] = x[1] / len(data1)
    counts[j] = x
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


print(X1_test.iloc[0:10][[' Source IP', ' Destination IP', 'label']])
print("=========================================================================")
# modifs
src_ip = X1_test[' Source IP']
dst_ip = X1_test[' Destination IP']
data_labels = X1_test['label']

columnss = list(set(list(X1_test.iloc[:, :].columns )) - set(list([' Source IP', ' Destination IP', 'label'])))

X1_test = X1_test[columnss]

func(X1_test)