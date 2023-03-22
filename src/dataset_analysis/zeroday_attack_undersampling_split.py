''''
    We need to implement tghe undersampling this time by keeping the xp on Monday and Tuesday
'''

import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

monday_file = "./input/Dataset/Monday-WorkingHours.pcap_ISCX.csv"
path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

X_Train = pd.DataFrame()
X_Test = pd.DataFrame()

for i in range(file_count):
    data1 = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    if ("Tuesday" in files[i]) or ("Wednesday" in files[i]) :
        X_Train = pd.concat([X_Train, data1], ignore_index = True)
    else :
        X_Test = pd.concat([X_Test, data1], ignore_index = True)

# Add Monday to the train set
# data1 = pd.read_csv(f'{monday_file}', encoding="ISO-8859–1", dtype = str)
# X_Train = pd.concat([X_Train, data1], ignore_index = True)




# Apply undersampling on the Train set
# Shuffle the Dataset.
X_Train = shuffle(X_Train)

# Put all the ATTACKs class in a separate dataset.
attacks_df = X_Train.loc[X_Train[' Label'] != 'BENIGN']

# Randomly select len(attacks_df) observations from the BENIGN (majority class)
benign_df = X_Train.loc[X_Train[' Label'] == 'BENIGN'].sample(n=len(attacks_df),random_state=42)

# Concatenate both dataframes again
X_Train = pd.concat([attacks_df, benign_df])


# At this step we have a balanced Train set


print("Conacat DONE -> Now shuffle the rows of both Train and Test sets")

# Shuffle both Train and Test sets
X_Train = shuffle(X_Train)
X_Test = shuffle(X_Test)
print("Shuffle done -> Split both Train and Test sets into Mega batchs")

# Save the Train set data file
X_Train.to_csv(f'./input/Dataset/ZeroDayAttacks_Split/Train/Train.csv', sep=',', index = False)

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
