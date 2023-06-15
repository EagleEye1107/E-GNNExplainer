import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


path, dirs, files = next(os.walk("./input/Dataset/GlobalDataset/Splitted/"))
file_count = len(files)

data_g = pd.DataFrame()
for nb_files in range(file_count):
    data1 = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)
    data_g = pd.concat([data_g, data1], ignore_index = True)
    
print(data_g)


a = data_g[' Source IP'].value_counts()[0:15]
b = []
for x in a.keys():
    b.append(a.get(key = x))

objects = list(a.keys())
y_pos = np.arange(len(objects))
performance = b


# plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(30, 10))
barlist = ax.bar(y_pos, performance, align='center', color = "orange")
barlist[0].set_color('r')
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()