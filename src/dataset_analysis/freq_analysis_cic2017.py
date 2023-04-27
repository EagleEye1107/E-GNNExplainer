import pandas as pd
import os

path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

print()

for i in range(file_count):
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    ##################### LABELS FREQ #######################################
    print()
    print("labels freq after changing labels to binary")
    counts = list(data1[' Label'].value_counts().to_dict().items())
    for j, x in enumerate(counts):
        x = list(x)
        x[1] = x[1] / len(data1)
        counts[j] = x
    print({f'{files[i]}' : counts})
    ##############################################################################


data1 = pd.read_csv('./input/Dataset/Monday-WorkingHours.pcap_ISCX.csv', encoding="ISO-8859–1", dtype = str)
##################### LABELS FREQ #######################################
print()
print("labels freq after changing labels to binary")
counts = list(data1[' Label'].value_counts().to_dict().items())
for j, x in enumerate(counts):
    x = list(x)
    x[1] = x[1] / len(data1)
    counts[j] = x
print({'Monday-WorkingHours.pcap_ISCX.csv' : counts})
##############################################################################