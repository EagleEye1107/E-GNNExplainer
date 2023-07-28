import os
import pandas as pd
import numpy as np

freq_list = []
binary_freq_list = []

path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
# path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted_With_Monday/"))
file_count = len(files)

for i in range(file_count):
    print(f'{files[i]} read !')
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    counts = list(data1[' Label'].value_counts().to_dict().items())
    for j, x in enumerate(counts):
        x = list(x)
        x[1] = x[1] / len(data1)
        counts[j] = x
    freq_list.append({f'{files[i]}' : counts})

print()
print(freq_list[0])
print()
print(freq_list[1])
print()
print(freq_list[2])
print()
print(freq_list[3])
print()

i = 0

for file_freq in freq_list:
    binary_freq_sublist = []
    other_labels = ['Other', 0]
    for label_freq in file_freq:
        if label_freq[0] == 'BENIGN':
            binary_freq_sublist.append(label_freq)
        else:
            other_labels[1] += label_freq[1]
    binary_freq_sublist.append(other_labels)
    binary_freq_list.append({f'file_{i}' : binary_freq_sublist})
    i += 1

print(binary_freq_list)
