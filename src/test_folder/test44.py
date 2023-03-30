import os
import pandas as pd


from_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset3.csv"
to_file = "./input/Dataset/GlobalDataset/Splitted/CIC-IDS-2017-Dataset3.csv"


data3 = pd.read_csv(f'{from_file}', encoding="ISO-8859–1", dtype = str)
data3 = data3.loc[data3[' Label'] == "Heartbleed"].head(3)

data2 = pd.read_csv(f'{to_file}', encoding="ISO-8859–1", dtype = str)
data2 = pd.concat([data2, data3], ignore_index = True)