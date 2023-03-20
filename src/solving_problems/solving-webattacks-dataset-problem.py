import pandas as pd

data1 = pd.read_csv('./input/Dataset/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', encoding="ISO-8859–1", dtype = str)
data1.drop(range(170366, len(data1.values)), inplace = True)
data1.to_csv("./input/Dataset/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", sep=',', index = False)
