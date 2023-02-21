import pandas as pd
import os
import numpy as np

path, dirs, files = next(os.walk("./Dataset/MachineLearningCVE/"))
file_count = len(files)

print()

for i in range(file_count):
    df = pd.read_csv(f'{path}{files[i]}', encoding="ISO-8859–1", dtype = str)
    print(f'''{files[i]} contains a duplicate in : 
    {df[df.duplicated(subset=df.columns.values, keep=False)]}''')