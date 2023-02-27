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

X_dataset.to_csv("./input/Dataset/GlobalDataset/CIC-IDS-2017-Dataset.csv", sep=',', index = False)

print("DONE !")