import os
import pandas as pd
from sklearn.utils import shuffle

path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

X_dataset = pd.DataFrame()

for i in range(file_count):
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    X_dataset = pd.concat([X_dataset, data1], ignore_index = True)
    print(f'file {i} named {files[i]} -> DONE')

print("Conacat DONE -> Now shuffle the rows")
X_dataset = shuffle(X_dataset)
print("Shuffle done -> save csv file")

j = a = b = 0
# Without Monday dataset file (containing only BENIGN instances) / split dataset into 4 dataset files
for i in range(len(X_dataset)):
    if i != 0 :
        if i % int(len(X_dataset)/5) == 0:
            # print(i)
            a = b
            b = i
            if b >= ((4/5) * len(X_dataset)) :
                b = len(X_dataset)
            print(f"[{a}, {b}]")
            df = X_dataset.iloc[a:b]


            # Add to Least populated class if nb occurences < 2
            print(df[" Label"].value_counts()[-1])
            print(df[" Label"].value_counts().index[-1])
            if df[" Label"].value_counts()[-1] < 2 :
                df_search = X_dataset.iloc[b:len(X_dataset)]
                needed_instances = df_search.loc[df_search[' Label'] == df[" Label"].value_counts().index[-1]].head(2 - df[" Label"].value_counts()[-1])
                df = pd.concat([df, needed_instances], ignore_index = True)
                # Delete the added rows from X_dataset.iloc[b:len(X_dataset)]
                
            df.to_csv(f'./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset{j}.csv', sep=',', index = False)
            j += 1

print("DONE !")