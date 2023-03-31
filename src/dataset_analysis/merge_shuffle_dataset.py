import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Classes
clss = ['BENIGN', 'Brute Force', 'XSS', 'Sql Injection', 'Heartbleed', 'DoS Hulk', 'DDoS', 'PortScan', 'FTP-Patator', 'Bot', 'DoS slowloris', 'DoS GoldenEye', 'DoS Slowhttptest', 'SSH-Patator', 'Infiltration']

path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

X_dataset = pd.DataFrame()

for i in range(file_count):
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    X_dataset = pd.concat([X_dataset, data1], ignore_index = True)
    print(f'file {i} named {files[i]} -> DONE')

# With Monday dataset file (containing only BENIGN instances)
# data1 = pd.read_csv(f'./input/Dataset/Monday-WorkingHours.pcap_ISCX.csv', encoding="ISO-8859–1", dtype = str)
# X_dataset = pd.concat([X_dataset, data1], ignore_index = True)
# print(f'file {i+1} named Monday-WorkingHours.pcap_ISCX.csv -> DONE')

print("Conacat DONE -> Now shuffle the rows")
X_dataset = shuffle(X_dataset)
print("Shuffle done -> save csv file")

j = a = b = 0
# Split dataset into 5 dataset files
print("len(X_dataset) = ", len(X_dataset))

# Class mapping abreviation
X_dataset[" Label"] = np.where(X_dataset[" Label"].str.contains('Brute Force'), 'Brute Force', X_dataset[" Label"])
X_dataset[" Label"] = np.where(X_dataset[" Label"].str.contains('XSS'), 'XSS', X_dataset[" Label"])
X_dataset[" Label"] = np.where(X_dataset[" Label"].str.contains('Sql Injection'), 'Sql Injection', X_dataset[" Label"])

# print(X_dataset[" Label"].value_counts())
# print(dfghjkl)

'''
for i in range(len(X_dataset) + 1):
    if i != 0 :
        if i % int(len(X_dataset)/5) == 0:
            a = b
            b = i
            print(f"[{a}, {b}]")
            df = X_dataset.iloc[a:b]
            # Add to Least populated class if nb occurences < 2
            # print(df[" Label"].value_counts()[-1])
            # print(df[" Label"].value_counts().index[-1])
            if df[" Label"].value_counts()[-1] < 2 :
                df_search = X_dataset.iloc[b:len(X_dataset)]
                needed_instances = df_search.loc[df_search[' Label'] == df[" Label"].value_counts().index[-1]].head(2 - df[" Label"].value_counts()[-1])
                df = pd.concat([df, needed_instances], ignore_index = True)
                # Delete the added rows from X_dataset.iloc[b:len(X_dataset)]
            df.to_csv(f'./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset{j}.csv', sep=',', index = False)
            j += 1

'''

# Split dataset into 8 dataset files
for i in range(1, len(X_dataset) + 1):
    if i % int(len(X_dataset)/5) == 0:
        # print(i)
        a = b
        b = i
        if b >= ((4/5) * len(X_dataset)) :
            b = len(X_dataset)
        print(f"[{a}, {b}]")
        df = X_dataset.iloc[a:b]

        # Delete Heartbleed instances to add them manually
        print("j = ", j)
        if j != 4 :
            df = df.drop(df.loc[df[' Label'] == 'Heartbleed'].index)
            inst = X_dataset.loc[X_dataset[' Label'] == 'Heartbleed'].head(2)
            df = pd.concat([df, inst], ignore_index = True)
            X_dataset = X_dataset.drop(inst.index)
            b -= 2
        print("Added Heartbleed instances")
        print("nb occ left of Heartbleed in X_dataset = ", len(X_dataset.loc[X_dataset[' Label'] == "Heartbleed"]))

        # IF we have a missing class in the datafile
        if len(df[" Label"].value_counts()) < 15 :
            # There is a class that doesn't figure in the datafile
            for cls in clss :
                if cls not in df[" Label"].value_counts().index:
                    print(f"{cls} not found")
                    # Add 2 intances to the DF and delete them from the dataset so we won't have possible duplicates
                    inst = X_dataset.loc[X_dataset[' Label'] == cls].tail(2)
                    df = pd.concat([df, inst], ignore_index = True)
                    X_dataset = X_dataset.drop(inst.index)
                    print(f"{cls} added")
                    print("len(X_dataset) : ", len(X_dataset))

        # IF we have a class with less than 2 occurrences
        least_pop_clss = df[' Label'].value_counts().index[-1]
        if df[" Label"].value_counts()[-1] < 2 :
            print(f"{least_pop_clss} occ is < 2")
            inst = X_dataset.loc[X_dataset[' Label'] == df[" Label"].value_counts().index[-1]].tail(1)
            df = pd.concat([df, inst], ignore_index = True)
            X_dataset = X_dataset.drop(inst.index)
            print(f"{least_pop_clss} occ completed")
            print("len(X_dataset) : ", len(X_dataset))

        df.to_csv(f'./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset{j}.csv', sep=',', index = False)
        j += 1


print("DONE !")
