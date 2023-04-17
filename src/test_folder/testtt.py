import pandas as pd
import os

path, dirs, files = next(os.walk("./input/Dataset/TrafficLabelling/"))
file_count = len(files)

labels = []
att_count_glb = []
att_glb = []

attt = []

print()

for i in range(file_count):
    df = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    df_values = df.values

    # Retrieve all existing labels in the file
    label_column = set(df_values[:, -1])
    labels += label_column

    # print(df[" Label"].value_counts())

    # Attributes count
    att_count = len(df_values[0, :]) - 1
    att_count_glb.append(att_count)

    # Iterate all the attributes to check if they are similar in each Dataset file
    att_glb += list(df.columns.values)
    attt = df.columns.values

    # Print Results
    print(f'{files[i]} contains -> {att_count} attribute and labels are [ {df[" Label"].value_counts()} ]')


print()
labels = set(labels)
att_count_glb = list(set(att_count_glb))
att_glb = list(set(att_glb))

if len(att_count_glb) > 1: 
    print('Error : number of attributes is different in the Dataset files !')
elif len(att_glb) - 1 != att_count_glb[0] :
    # -1 so we don't consider the label column as an attribute
    print('Error : Dataset files doesnt have the same attributes !')
else :
    print(f'''In the CIC-IDS-2017 We have :
            -> {att_count_glb[0]} attribute, all similar in all the dataset files
            -> Attributes are : {attt[0:-1]}
            -> {len(labels)} labels
            -> Labels are : {labels}''')

print()
