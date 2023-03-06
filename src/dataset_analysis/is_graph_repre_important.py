'''
    The idea here is to change all nodes to a single node,
    With this modification we won't have the graph structure,
    This approach is to understand wethere or not the graph structure have a good impact on the classification or not
'''


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

# With Monday dataset file (containing only BENIGN instances)
# data1 = pd.read_csv(f'./input/Dataset/Monday-WorkingHours.pcap_ISCX.csv', encoding="ISO-8859–1", dtype = str)
# X_dataset = pd.concat([X_dataset, data1], ignore_index = True)
# print(f'file {i+1} named Monday-WorkingHours.pcap_ISCX.csv -> DONE')

print("Conacat DONE -> Now shuffle the rows")
X_dataset = shuffle(X_dataset)
print("Shuffle done -> Modify IPAdr")


src_ip = list(range(len(X_dataset.values)))
dest_ip = list(range(len(X_dataset.values), 2*len(X_dataset.values)))

X_dataset[' Source IP'] = src_ip
X_dataset[' Destination IP'] = dest_ip

print(X_dataset)

print("Modify IPAdr done -> Save csv file")

j = a = b = 0
# Without Monday dataset file (containing only BENIGN instances) / split dataset into 4 dataset files
for i in range(len(X_dataset)):
    if i != 0 :
        if i % int(len(X_dataset)/4) == 0:
            # print(i)
            a = b
            b = i
            if b >= ((3/4) * len(X_dataset)) :
                b = len(X_dataset)
            print(f"[{a}, {b}]")
            df = X_dataset.iloc[a:b]
            df.to_csv(f'./input/Dataset/GlobalDataset/Splitted_Modified_IP/CIC-IDS-2017-Dataset{j}.csv', sep=',', index = False)
            j += 1

# With Monday dataset file (containing only BENIGN instances) / split dataset into 5 dataset files
# for i in range(len(X_dataset)):
#     if i != 0 :
#         if i % int(len(X_dataset)/5) == 0:
#             # print(i)
#             a = b
#             b = i
#             if b >= ((4/5) * len(X_dataset)) :
#                 b = len(X_dataset)
#             print(f"[{a}, {b}]")
#             df = X_dataset.iloc[a:b]
#             df.to_csv(f'./input/Dataset/GlobalDataset/Splitted_With_Monday/CIC-IDS-2017-Dataset{j}.csv', sep=',', index = False)
#             j += 1



# X_dataset.to_csv("./input/Dataset/GlobalDataset/CIC-IDS-2017-Dataset.csv", sep=',', index = False)

print("DONE !")