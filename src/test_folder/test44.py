import pandas as pd



# 2 -> 4 : 1 Heartbleed
from_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset2.csv"
to_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset4.csv"

data_from1 = pd.read_csv(f'{from_file}', encoding="ISO-8859–1", dtype = str)
data_to = pd.read_csv(f'{to_file}', encoding="ISO-8859–1", dtype = str)

data_from = data_from1.loc[data_from1[' Label'] == "Heartbleed"].head(1)
data_to = pd.concat([data_to, data_from], ignore_index = True)
data_from1.drop(data_from.index, inplace = True)

print(data_from1[" Label"].value_counts())
print(data_to[" Label"].value_counts())

data_from1.to_csv(from_file, sep=',', index = False)
data_to.to_csv(to_file, sep=',', index = False)



# 0 -> 1 : 2 Heartbleed
from_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset0.csv"
to_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset1.csv"

data_from1 = pd.read_csv(f'{from_file}', encoding="ISO-8859–1", dtype = str)
data_to = pd.read_csv(f'{to_file}', encoding="ISO-8859–1", dtype = str)

data_from = data_from1.loc[data_from1[' Label'] == "Heartbleed"].head(2)
data_to = pd.concat([data_to, data_from], ignore_index = True)
data_from1.drop(data_from.index, inplace = True)

print(data_from1[" Label"].value_counts())
print(data_to[" Label"].value_counts())

data_from1.to_csv(from_file, sep=',', index = False)
data_to.to_csv(to_file, sep=',', index = False)
