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

print("Conacat DONE -> Now shuffle the rows and do UNDERSAMPLING")
X_dataset = shuffle(X_dataset)

# Put all the ATTACKs class in a separate dataset.
attacks_df = X_dataset.loc[X_dataset[' Label'] != 'BENIGN']

# Randomly select len(attacks_df) observations from the BENIGN (majority class)
benign_df = X_dataset.loc[X_dataset[' Label'] == 'BENIGN'].sample(n=len(attacks_df),random_state=42)

# Concatenate both dataframes again
X_dataset = pd.concat([attacks_df, benign_df])

# At this step we have a balanced Train set
X_dataset = shuffle(X_dataset)
print("Shuffle done -> save csv file")

print(len(X_dataset.values))

X_dataset.to_csv("./input/Dataset/GlobalDataset/CIC-IDS-2017-Dataset.csv", sep=',', index = False)

print("DONE !")