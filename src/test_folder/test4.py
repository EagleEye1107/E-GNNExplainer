import pandas as pd
import os


data1 = pd.read_csv('./input/Dataset/GlobalDataset/CIC-IDS-2017-Dataset.csv')

print(len(data1.values))


path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted_Modified_IP/"))
file_count = len(files)

cpt = 0

for i in range(file_count):
    print(i)
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    cpt += len(data1.values)

    src_ip = list(range(len(data1.values)))
    dest_ip = list(range(len(data1.values), 2*len(data1.values)))

    print(src_ip[-1])
    print(dest_ip[-1])

    data1[' Source IP'] = src_ip
    data1[' Destination IP'] = dest_ip
    print(data1)

print(cpt)