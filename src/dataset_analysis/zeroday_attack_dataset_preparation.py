import os
import pandas as pd
from sklearn.utils import shuffle

monday_file = "./input/Dataset/Monday-WorkingHours.pcap_ISCX.csv"
path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

X_Train = pd.DataFrame()
X_Test = pd.DataFrame()

for i in range(file_count):
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    if ("Tuesday" in files[i]) : 
        # or ("Wednesday" in files[i]) :
        X_Train = pd.concat([X_Train, data1], ignore_index = True)
    else :
        X_Test = pd.concat([X_Test, data1], ignore_index = True)

# Add Monday to the train set
data1 = pd.read_csv(f'{monday_file}', encoding="ISO-8859–1", dtype = str)
X_Train = pd.concat([X_Train, data1], ignore_index = True)

print("Conacat DONE -> Now shuffle the rows of both Train and Test sets")

# Shuffle both Train and Test sets
X_Train = shuffle(X_Train)
X_Test = shuffle(X_Test)
print("Shuffle done -> Split both Train and Test sets into Mega batchs")


j = a = b = 0
# Split Train set into 5 Training sets
for i in range(len(X_Train)):
    if i != 0 :
        if i % int(len(X_Train)/5) == 0:
            a = b
            b = i
            if b >= ((4/5) * len(X_Train)) :
                b = len(X_Train)
            print(f"[{a}, {b}]")
            df = X_Train.iloc[a:b]
            df.to_csv(f'./input/Dataset/ZeroDayAttacks_Split/Train/Train{j}.csv', sep=',', index = False)
            j += 1


j = a = b = 0
# Split Test set into 5 Testing sets
for i in range(len(X_Test)):
    if i != 0 :
        if i % int(len(X_Test)/5) == 0:
            a = b
            b = i
            if b >= ((4/5) * len(X_Test)) :
                b = len(X_Test)
            print(f"[{a}, {b}]")
            df = X_Test.iloc[a:b]
            df.to_csv(f'./input/Dataset/ZeroDayAttacks_Split/Test/Test{j}.csv', sep=',', index = False)
            j += 1


print("DONE !")
